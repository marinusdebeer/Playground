/* eslint-disable react-native/no-inline-styles */
import React, {useState} from 'react';
import EventSource from 'react-native-event-source';
import {Configuration, OpenAIApi} from 'openai';
import 'react-native-url-polyfill/auto';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  useColorScheme,
  View,
  Button,
  TouchableOpacity,
  Text,
  Image,
  TextInput,
} from 'react-native';

import {Colors, Header} from 'react-native/Libraries/NewAppScreen';
const API_KEY = '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';
const configuration = new Configuration({
  apiKey: API_KEY,
});
const openai = new OpenAIApi(configuration);
function App(): JSX.Element {
  let [imageUrl, setImageUrl] = useState<string | null>(null);
  let [chatText, setChatText] = useState<string | null>(null);
  let [conversation, setConversation] = useState<string[]>([]);
  let [prompt, setPrompt] = useState<string>('');
  const isDarkMode = useColorScheme() === 'dark';

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };

  function createStream() {
    const url = 'https://api.openai.com/v1/chat/completions/sse'; // Replace with your SSE server URL

    const eventSource = new EventSource(url);

    eventSource.onmessage = event => {
      console.log('Received data:', event.data);
      // Process the streamed data here
    };

    eventSource.addEventListener('open', event => {
      console.log('Connection opened:', event);
    });

    eventSource.addEventListener('error', error => {
      if (error.eventPhase === EventSource.CLOSED) {
        console.log('Connection closed');
      } else {
        console.error('Error:', error);
      }
    });

    // To close the stream when you're done
    // eventSource.close();
  }
  createStream();
  async function chat() {
    console.log('Chat started');
    const response = await openai.createChatCompletion({
      messages: [{role: 'user', content: prompt}],
      model: 'gpt-3.5-turbo',
      max_tokens: 200,
      // stream: true,
    });
    // console.log(response);
    console.log(response.data.choices[0].message.content);
    setChatText(response.data.choices[0].message.content);
    setConversation([
      ...conversation,
      response.data.choices[0].message.content,
    ]);
    console.log(conversation);
    /* await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        messages: [{role: 'user', content: prompt}],
        max_tokens: 200,
        model: 'gpt-3.5-turbo',
        n: 1,
        stop: null,
        stream: true,
        temperature: 1,
      }),
    })
      .then(response => {
        console.log('1', response);
        // return response.json();
      })
      .then(data => {
        // console.log(data);
        // console.log(data.choices[0].message.content);
        // setChatText(data.choices[0].message.content);
      }); */
  }
  async function image() {
    console.log('Image generation started');
    try {
      const response = await openai.createImage({
        prompt: prompt,
        n: 1,
        size: '512x512',
      });
      console.log(response.data.data[0].url);
      console.log(response.data.data[0].url);
      setImageUrl(response.data.data[0].url);
      /* await fetch('https://api.openai.com/v1/images/generations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`,
        },
        body: JSON.stringify({
          model: 'image-alpha-001',
          prompt: prompt,
        }),
      })
        .then(response => {
          return response.json();
        })
        .then(data => {
          console.log(data.data[0].url);
          setImageUrl(data.data[0].url);
          // imageUrl = data.data[0].url;
        }); */
    } catch (error) {
      console.error('error: ', error);
    }
  }
  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={backgroundStyle}>
        {/* <Header /> */}
        <View
          style={{
            backgroundColor: isDarkMode ? Colors.black : Colors.white,
            alignItems: 'center',
            justifyContent: 'center',
            padding: 10,
          }}>
          <Text style={{fontSize: 20, fontWeight: 'bold', margin: 10}}>
            GPT-3 Mobile App
          </Text>
          
          </View>
          {conversation.map((message, index) => (
          <ScrollView 
            key={index}
            style={{
              height: 300,
              margin: 20,
              width: 350,
              borderWidth: 1,
              borderColor: 'gray',
              borderRadius: 10,
            }}>
            <Text style={{fontSize: 18, margin: 20}}>{message}</Text>
          </ScrollView>
        ))}
        <ScrollView
            style={{
              margin: 10,
              borderWidth: 1,
              padding: 10,
              borderRadius: 10,
              borderColor: 'gray',
              width: 350,
            }}>
            <TextInput
              style={{
                height: 80,
              }}
              placeholder="Prompt here"
              multiline={true}
              onChangeText={newText => setPrompt(newText)}
            />
          </ScrollView>
          <View style={{flexDirection: 'row'}}>
            <TouchableOpacity
              style={{
                backgroundColor: '#ff8c00',
                borderRadius: 10,
                height: 50,
                width: 120,
                alignItems: 'center',
                justifyContent: 'center',
                margin: 10,
              }}
              onPress={chat}>
              <Text style={{color: 'white', fontSize: 20}}>Chat</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={{
                backgroundColor: '#4b0082',
                borderRadius: 10,
                height: 50,
                width: 120,
                alignItems: 'center',
                justifyContent: 'center',
                margin: 10,
              }}
              onPress={image}>
              <Text style={{color: 'white', fontSize: 20}}>Image</Text>
            </TouchableOpacity>
          <ScrollView
            style={{
              height: 300,
              margin: 20,
              width: 350,
              borderWidth: 1,
              borderColor: 'gray',
              borderRadius: 10,
            }}>
            <Text style={{fontSize: 18, margin: 20}}>{chatText}</Text>
          </ScrollView>
          <Image
            source={{
              uri: imageUrl,
              width: 200,
              height: 200,
            }}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
export default App;
