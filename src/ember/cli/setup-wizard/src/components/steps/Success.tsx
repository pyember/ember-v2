import React from 'react';
import {Box, Text} from 'ink';
import {Provider, PROVIDERS} from '../../types.js';

interface Props {
  provider: Provider;
  onComplete: () => void;
}

export const Success: React.FC<Props> = ({provider, onComplete}) => {
  const providerInfo = PROVIDERS[provider];

  React.useEffect(() => {
    const timer = setTimeout(onComplete, 5000);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <Box flexDirection="column">
      <Box marginBottom={2}>
        <Text color="green" bold>
          ✨ Setup complete!
        </Text>
      </Box>

      <Box flexDirection="column" gap={1}>
        <Box>
          <Text color="green">✓</Text>
          <Text> API key configured for {providerInfo.name}</Text>
        </Box>
        
        <Box>
          <Text color="green">✓</Text>
          <Text> Example created: hello_ember.py</Text>
        </Box>
        
        <Box>
          <Text color="green">✓</Text>
          <Text> Configuration saved to ~/.ember/config.json</Text>
        </Box>
      </Box>

      <Box marginTop={2} borderStyle="round" borderColor="cyan" paddingX={2} paddingY={1}>
        <Box flexDirection="column">
          <Text bold>Quick Start:</Text>
          <Box marginTop={1}>
            <Text color="cyan">$ python hello_ember.py</Text>
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
          Happy building! 🚀
        </Text>
      </Box>
    </Box>
  );
};