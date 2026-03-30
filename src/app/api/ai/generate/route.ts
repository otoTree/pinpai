
import { NextResponse } from 'next/server';
import {
  getSystemPrompt,
  getProjectBlueprintPrompt,
  getStoryBatchPrompt,
  getEpisodeContentPrompt,
  getProjectDetailsPrompt,
} from '@/lib/prompts';
import { createClient } from '@/lib/supabase/server';
import { AIAPIError, callAIChatCompletion, extractFirstMessageContent } from '@/lib/ai-server';

export const maxDuration = 300; // Allow longer timeout for generation

const parseJSONFromLLM = (content: string) => {
  try {
    return JSON.parse(content);
  } catch {
    const firstBrace = content.indexOf('{');
    const lastBrace = content.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(content.slice(firstBrace, lastBrace + 1));
    }
    throw new Error('Invalid JSON response');
  }
};

const normalizeAssetName = (value: string) => value.trim().toLowerCase();
const toErrorMessage = (error: unknown) => error instanceof Error ? error.message : 'Internal Server Error';
const parseErrorDetails = (details?: string) => {
  if (!details) return null;
  try {
    return JSON.parse(details);
  } catch {
    return details;
  }
};

const creativeGenerationTypes = new Set(['story_blueprint', 'story_batch']);

const buildJsonOnlyPrompt = (prompt: string) => `${prompt}

Critical Output Rule:
- Return exactly one valid JSON object.
- Do not use markdown code fences.
- Do not add explanations before or after the JSON.
- Ensure all required fields from the schema are present.`;

const buildRetryJsonPrompt = (prompt: string) => `${buildJsonOnlyPrompt(prompt)}
- Keep every string concise.
- Do not truncate the response.
- Complete the entire JSON before stopping.`;

const generateJsonContent = async ({
  type,
  targetLanguage,
  prompt,
  attempt = 'primary',
}: {
  type: string;
  targetLanguage: string;
  prompt: string;
  attempt?: 'primary' | 'retry';
}) => {
  const isCreativeGeneration = creativeGenerationTypes.has(type);
  const maxTokens =
    type === 'story_blueprint'
      ? 3200
      : type === 'story_batch'
        ? attempt === 'retry'
          ? 4200
          : 3600
        : type === 'episode'
          ? 4000
          : undefined;

  const runCompletion = async (mode: 'structured' | 'plain') => {
    const data = await callAIChatCompletion({
      messages: [
        { role: 'system', content: getSystemPrompt(targetLanguage) },
        {
          role: 'user',
          content: mode === 'plain'
            ? (attempt === 'retry' ? buildRetryJsonPrompt(prompt) : buildJsonOnlyPrompt(prompt))
            : prompt,
        },
      ],
      temperature: isCreativeGeneration
        ? (attempt === 'retry' ? 0.4 : 0.6)
        : 0.7,
      maxTokens,
      extraPayload: mode === 'structured' ? { response_format: { type: 'json_object' } } : undefined,
    });

    return extractFirstMessageContent(data);
  };

  if (isCreativeGeneration) {
    try {
      return await runCompletion('plain');
    } catch (error) {
      if (!(error instanceof AIAPIError) || error.status < 500) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 800));
      return await runCompletion('plain');
    }
  }

  return await runCompletion('structured');
};

export async function POST(req: Request) {
  try {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const {
      type,
      theme,
      series_plan,
      episode_num,
      summary,
      existing_assets,
      language,
      episode_count,
      start_episode,
      end_episode,
      project_blueprint,
      story_analysis,
      existing_episodes,
      product_asset_details,
    } = await req.json();
    const targetLanguage = language || 'zh';

    let prompt = '';
    let stage = type || 'unknown';

    if (type === 'story_blueprint') {
      prompt = getProjectBlueprintPrompt(theme, targetLanguage, episode_count, typeof product_asset_details === 'string' ? product_asset_details : '');
    } else if (type === 'story_batch') {
      prompt = getStoryBatchPrompt(
        theme,
        targetLanguage,
        episode_count,
        start_episode,
        end_episode,
        project_blueprint,
        story_analysis,
        existing_episodes
      );
    } else if (type === 'episode') {
      prompt = getEpisodeContentPrompt(
        episode_num,
        series_plan,
        summary,
        targetLanguage,
        Array.isArray(existing_assets) ? existing_assets : []
      );
    } else if (type === 'project_details') {
      prompt = getProjectDetailsPrompt(theme);
    } else {
      return NextResponse.json({ error: 'Invalid type' }, { status: 400 });
    }

    let content = '';
    try {
      content = await generateJsonContent({
        type,
        targetLanguage,
        prompt,
      });
    } catch (error) {
      if (error instanceof AIAPIError) {
        return NextResponse.json(
          {
            error: error.message,
            details: parseErrorDetails(error.details),
            type,
            stage,
            upstreamStatus: error.status,
          },
          { status: error.status }
        );
      }
      throw error;
    }

    try {
      const jsonContent = parseJSONFromLLM(content);
      if (type === 'episode' && Array.isArray(existing_assets)) {
        const characterSet = new Set(
          existing_assets
            .filter((asset) => asset?.type === 'character' && typeof asset?.name === 'string')
            .map((asset) => normalizeAssetName(asset.name))
        );
        const locationSet = new Set(
          existing_assets
            .filter((asset) => asset?.type === 'location' && typeof asset?.name === 'string')
            .map((asset) => normalizeAssetName(asset.name))
        );

        const usedCharacters = Array.isArray(jsonContent.used_characters)
          ? jsonContent.used_characters.filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
          : [];
        const usedLocations = Array.isArray(jsonContent.used_locations)
          ? jsonContent.used_locations.filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
          : [];

        if (usedCharacters.length === 0 && characterSet.size > 0) {
          return NextResponse.json({ error: 'Invalid model output: missing used_characters' }, { status: 502 });
        }
        if (usedLocations.length === 0 && locationSet.size > 0) {
          return NextResponse.json({ error: 'Invalid model output: missing used_locations' }, { status: 502 });
        }

        const invalidCharacters = usedCharacters.filter((name: string) => !characterSet.has(normalizeAssetName(name)));
        const invalidLocations = usedLocations.filter((name: string) => !locationSet.has(normalizeAssetName(name)));

        if (invalidCharacters.length > 0 || invalidLocations.length > 0) {
          return NextResponse.json(
            {
              error: 'Script violated asset constraints',
              invalidCharacters,
              invalidLocations,
            },
            { status: 422 }
          );
        }
      }
      return NextResponse.json(jsonContent);
    } catch (e) {
      if (creativeGenerationTypes.has(type)) {
        let retryContent = '';
        try {
          retryContent = await generateJsonContent({
            type,
            targetLanguage,
            prompt,
            attempt: 'retry',
          });
          const retryJsonContent = parseJSONFromLLM(retryContent);
          return NextResponse.json(retryJsonContent);
        } catch (retryError) {
          stage = `${type}_response_parse`;
          console.error('JSON Parse Retry Error:', retryError);
          if (retryError instanceof AIAPIError) {
            return NextResponse.json(
              {
                error: retryError.message,
                details: parseErrorDetails(retryError.details),
                type,
                stage: `${type}_retry`,
                upstreamStatus: retryError.status,
                raw: retryContent || content,
              },
              { status: retryError.status }
            );
          }
          return NextResponse.json(
            {
              error: 'Invalid model JSON output',
              details: toErrorMessage(retryError),
              raw: retryContent || content,
              type,
              stage,
            },
            { status: 502 }
          );
        }
      }
      stage = `${type}_response_parse`;
      console.error('JSON Parse Error:', e);
      return NextResponse.json(
        {
          error: 'Invalid model JSON output',
          details: toErrorMessage(e),
          raw: content,
          type,
          stage,
        },
        { status: 502 }
      );
    }

  } catch (error) {
    if (error instanceof AIAPIError) {
      return NextResponse.json(
        {
          error: error.message,
          details: parseErrorDetails(error.details),
          upstreamStatus: error.status,
        },
        { status: error.status }
      );
    }
    console.error('API Route Error:', error);
    return NextResponse.json(
      {
        error: 'Internal Server Error',
        details: toErrorMessage(error),
      },
      { status: 500 }
    );
  }
}
