import React, {useEffect} from 'react';
import {Box, Text} from 'ink';
import Gradient from 'ink-gradient';
import BigText from 'ink-big-text';

interface Props {
  onNext: () => void;
}

export const Welcome: React.FC<Props> = ({onNext}) => {
  useEffect(() => {
    const timer = setTimeout(onNext, 2000);
    return () => clearTimeout(timer);
  }, [onNext]);

  return (
    <Box flexDirection="column" alignItems="center" paddingY={2}>
      <Gradient name="rainbow">
        <BigText text="EMBER" font="chrome" />
      </Gradient>
      <Box marginY={1}>
        <Text>Let's get you set up in 60 seconds</Text>
      </Box>
      <Box marginTop={2}>
        <Text dimColor>Starting...</Text>
      </Box>
    </Box>
  );
};