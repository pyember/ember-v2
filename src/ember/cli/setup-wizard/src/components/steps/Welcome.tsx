import React, {useEffect} from 'react';
import {Box, Text} from 'ink';
import Gradient from 'ink-gradient';
import BigText from 'ink-big-text';
import {LogoDisplay} from '../LogoDisplay.js';

interface Props {
  onNext: () => void;
}

export const Welcome: React.FC<Props> = ({onNext}) => {
  useEffect(() => {
    const timer = setTimeout(onNext, 2000);
    return () => clearTimeout(timer);
  }, [onNext]);

  return (
    <Box flexDirection="column" alignItems="center" paddingY={3}>
      <Box marginBottom={2}>
        <LogoDisplay provider="ember" variant="ascii" size="large" />
      </Box>
      <Box marginBottom={1}>
        <Text color="blue" bold>
          EMBER
        </Text>
      </Box>
      <Box marginBottom={3}>
        <Text dimColor>
          The inference-time scaling architectures and compound agent systems framework
        </Text>
      </Box>
      <Box>
        <Text dimColor italic>
          Initializing setup...
        </Text>
      </Box>
    </Box>
  );
};