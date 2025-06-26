import React, {useState, useEffect} from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import {LogoDisplay} from '../LogoDisplay.js';
import {Provider, PROVIDERS} from '../../types.js';
import {checkCredentials} from '../../utils/config.js';

interface Props {
  onSelect: (provider: Provider) => void;
  onBack?: () => void;
  onCancel?: () => void;
}

interface ProviderStatus {
  configured: boolean;
  keyPreview?: string;
}

export const ProviderSelection: React.FC<Props> = ({onSelect, onBack, onCancel}) => {
  const [providerStatuses, setProviderStatuses] = useState<Record<Provider, ProviderStatus>>({} as Record<Provider, ProviderStatus>);

  useEffect(() => {
    // Check configuration status for each provider
    const checkStatuses = async () => {
      const statuses: Record<Provider, ProviderStatus> = {} as Record<Provider, ProviderStatus>;
      
      for (const provider of Object.keys(PROVIDERS) as Provider[]) {
        const result = await checkCredentials(provider);
        statuses[provider] = result;
      }
      
      setProviderStatuses(statuses);
    };
    
    checkStatuses();
  }, []);
  const items = Object.entries(PROVIDERS).map(([key, info]) => ({
    label: `${info.name} - ${info.description}`,
    value: key as Provider,
    color: info.color,
  }));

  // Handle keyboard shortcuts
  useInput((input, key) => {
    if (key.escape) {
      onCancel?.();
    } else if ((input === 'b' || input === 'B')) {
      onBack?.();
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Which AI provider would you like to use?
        </Text>
      </Box>
      
      {/* Clean provider list */}

      <Box marginTop={2}>
        <SelectInput
          items={items}
          onSelect={(item) => onSelect(item.value)}
          indicatorComponent={() => <Text> </Text>}
          itemComponent={({label, isSelected}) => {
            const item = items.find(i => i.label === label);
            const provider = PROVIDERS[item?.value as Provider];
            return (
              <Box paddingY={1}>
                <Box>
                  <Text color="green">{isSelected ? '▸' : ' '} </Text>
                  <LogoDisplay 
                    provider={provider.id} 
                    variant="symbol"
                    size="small"
                  />
                  <Text> </Text>
                  <Text color={isSelected ? 'white' : 'gray'} bold={isSelected}>
                    {provider.name}
                  </Text>
                  {providerStatuses[provider.id]?.configured && (
                    <Text color="green"> ✓ {providerStatuses[provider.id]?.keyPreview}</Text>
                  )}
                </Box>
              </Box>
            );
          }}
        />
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Box>
          <Text dimColor>↑↓</Text>
          <Text dimColor> - Navigate providers</Text>
        </Box>
        <Box>
          <Text dimColor>Enter</Text>
          <Text dimColor> - Select provider</Text>
        </Box>
        <Box>
          <Text dimColor>B</Text>
          <Text dimColor> - Back to setup mode</Text>
        </Box>
        <Box>
          <Text dimColor>ESC</Text>
          <Text dimColor> - Cancel setup</Text>
        </Box>
      </Box>
    </Box>
  );
};