import React from 'react';
import {Box, Text} from 'ink';
import {LogoDisplay} from '../LogoDisplay.js';
import {Provider, PROVIDERS} from '../../types.js';

interface Props {
  provider: Provider;
  onComplete: () => void;
  configuredProviders?: Set<Provider>;
}

export const Success: React.FC<Props> = ({provider, onComplete, configuredProviders}) => {
  const providerInfo = PROVIDERS[provider];
  const isMultiSetup = configuredProviders && configuredProviders.size > 1;

  React.useEffect(() => {
    const timer = setTimeout(onComplete, 5000);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <Box flexDirection="column">
      <Box marginBottom={2}>
        <Text color="green" bold>
          Setup Complete
        </Text>
      </Box>

      <Box flexDirection="column" gap={1}>
        {isMultiSetup ? (
          <Box flexDirection="column">
            <Box marginBottom={1}>
              <Text color="green">[OK]</Text>
              <Text> Configured {configuredProviders?.size} providers:</Text>
            </Box>
            {Array.from(configuredProviders || []).map(p => (
              <Box key={p} paddingLeft={2} gap={1}>
                <LogoDisplay provider={p} variant="symbol" />
                <Text>{PROVIDERS[p].name}</Text>
              </Box>
            ))}
          </Box>
        ) : (
          <Box>
            <Text color="green">[OK]</Text>
            <Text> API key configured for {providerInfo.name}</Text>
          </Box>
        )}
        
        <Box>
          <Text color="green">[OK]</Text>
          <Text> Example created: hello_ember.py</Text>
        </Box>
        
        <Box>
          <Text color="green">[OK]</Text>
          <Text> Configuration saved to ~/.ember/config.json</Text>
        </Box>
      </Box>

      <Box marginTop={2} borderStyle="round" borderColor="cyan" paddingX={2} paddingY={1}>
        <Box flexDirection="column">
          <Text bold>Quick Start:</Text>
          <Box marginTop={1}>
            <Text color="cyan">$ python hello_ember.py</Text>
          </Box>
          <Box>
            <Text dimColor>(Created in current directory)</Text>
          </Box>
        </Box>
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Text bold>Next steps:</Text>
        <Box paddingLeft={2} flexDirection="column">
          <Text>• Explore models: models.discover()</Text>
          <Text>• Read docs: https://ember.ai/docs</Text>
          <Text>• Join community: https://discord.gg/ember</Text>
        </Box>
      </Box>

      <Box marginTop={3}>
        <Text dimColor italic>
          Ready to build.
        </Text>
      </Box>
    </Box>
  );
};