import React from 'react';
import {Box, Text} from 'ink';
import SelectInput from 'ink-select-input';
import {Provider, PROVIDERS} from '../../types.js';

interface Props {
  onSelect: (provider: Provider) => void;
}

export const ProviderSelection: React.FC<Props> = ({onSelect}) => {
  const items = Object.entries(PROVIDERS).map(([key, info]) => ({
    label: `${info.icon} ${info.name} - ${info.description}`,
    value: key as Provider,
  }));

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Which AI provider would you like to use?
        </Text>
      </Box>
      
      <Box flexDirection="column" gap={1}>
        {Object.values(PROVIDERS).map((provider) => (
          <Box key={provider.id} paddingLeft={2}>
            <Text dimColor>
              {provider.icon} {provider.name}: {provider.models.join(', ')}
            </Text>
          </Box>
        ))}
      </Box>

      <Box marginTop={2}>
        <SelectInput
          items={items}
          onSelect={(item) => onSelect(item.value)}
          indicatorComponent={({isSelected}) => (
            <Text color="green">{isSelected ? '▸' : ' '}</Text>
          )}
        />
      </Box>

      <Box marginTop={2}>
        <Text dimColor>Use arrow keys to select, Enter to confirm</Text>
      </Box>
    </Box>
  );
};