import React from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  useColorScheme,
  View,
  Button,
} from 'react-native';

import {Colors, Header} from 'react-native/Libraries/NewAppScreen';
const API_KEY = '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';

async function chat() {
  console.log('Chat started');
  await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      messages: [{role: 'user', content: 'Write a story'}],
      max_tokens: 100,
      model: 'gpt-3.5-turbo',
      n: 1,
      stop: null,
      temperature: 1,
    }),
  })
    .then(response => {
      return response.json();
    })
    .then(data => {
      console.log(data.choices[0].message.content);
    });
}
async function image() {
  console.log('Image generation started');
  try {
    await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: 'image-alpha-001',
        prompt:
          'image of an astronaut walking through a field of sunflowers in a galaxy far far away with stars in the background',
      }),
    })
      .then(response => {
        return response.json();
      })
      .then(data => {
        console.log(data.data[0].url);
      });
  } catch (error) {
    console.error('error: ', error);
  }
}
function App(): JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };

  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={backgroundStyle}>
        <Header />
        <View
          style={{
            backgroundColor: isDarkMode ? Colors.black : Colors.white,
          }}>
          <Button title="Press me" onPress={chat} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
export default App;
