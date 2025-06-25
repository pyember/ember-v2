import React, {useState, useEffect} from 'react';
import {Box, Text, useInput} from 'ink';
import TextInput from 'ink-text-input';
import Link from 'ink-link';
import {Provider, PROVIDERS} from '../../types.js';
import clipboardy from 'clipboardy';

interface Props {
  provider: Provider;
  onSubmit: (apiKey: string) => void;
  error?: string;
}

export const ApiKeySetup: React.FC<Props> = ({provider, onSubmit, error}) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [clipboardStatus, setClipboardStatus] = useState('');
  const providerInfo = PROVIDERS[provider];

  // Handle keyboard input for Tab and clipboard paste
  useInput((input, key) => {
    if (key.tab) {
      // Toggle visibility
      setShowKey(prev => !prev);
    } else if (input === 'p' || input === 'P') {
      // Paste from clipboard
      handleClipboardPaste();
    }
  });

  const handleClipboardPaste = async () => {
    try {
      const text = await clipboardy.read();
      if (text && text.trim()) {
        setApiKey(text.trim());
        setClipboardStatus('✓ Pasted from clipboard');
        setTimeout(() => setClipboardStatus(''), 2000);
      }
    } catch (err) {
      setClipboardStatus('✗ Could not read clipboard');
      setTimeout(() => setClipboardStatus(''), 2000);
    }
  };

  const handleSubmit = () => {
    if (apiKey.trim()) {
      onSubmit(apiKey.trim());
    }
  };

  const maskApiKey = (key: string) => {
    if (!key || key.length < 8) return key;
    if (showKey) return key;
    return key.slice(0, 3) + '•'.repeat(key.length - 6) + key.slice(-3);
  };

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Configure your {providerInfo.name} API key
        </Text>
      </Box>

      <Box marginBottom={2}>
        <Text>Get your API key from: </Text>
        <Link url={providerInfo.keyUrl}>
          <Text color="blue" underline>
            {providerInfo.keyUrl}
          </Text>
        </Link>
      </Box>

      {error && (
        <Box marginBottom={1}>
          <Text color="red">❌ {error}</Text>
        </Box>
      )}

      <Box>
        <Text>API Key: </Text>
        <Box flexDirection="column">
          <TextInput
            value={apiKey}
            onChange={setApiKey}
            onSubmit={handleSubmit}
            placeholder="Paste your API key here..."
            mask={showKey ? undefined : '•'}
          />
          <Box marginTop={1}>
            <Text dimColor>
              {showKey ? 'Visible' : 'Hidden'}: {maskApiKey(apiKey)}
            </Text>
          </Box>
          {clipboardStatus && (
            <Box marginTop={1}>
              <Text color="green">{clipboardStatus}</Text>
            </Box>
          )}
        </Box>
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Box>
          <Text dimColor>Tab</Text>
          <Text dimColor> - Toggle visibility ({showKey ? 'visible' : 'hidden'})</Text>
        </Box>
        <Box>
          <Text dimColor>P</Text>
          <Text dimColor> - Paste from clipboard</Text>
        </Box>
        <Box>
          <Text dimColor>Enter</Text>
          <Text dimColor> - Continue</Text>
        </Box>
      </Box>

      <Box marginTop={1}>
        <Text dimColor italic>
          Your key will be saved securely to ~/.ember/credentials
        </Text>
      </Box>
    </Box>
  );
};