/* eslint-disable react-native/no-inline-styles */
import React, {useState, useRef, useEffect} from 'react';
import EventSource from 'react-native-event-source';
import {Configuration, OpenAIApi} from 'openai';
import 'react-native-url-polyfill/auto';
// import {NavigationContainer} from '@react-navigation/native';
// import {createDrawerNavigator} from '@react-navigation/drawer';
// import HomeScreen from './Screens/HomeScreen';
// import SettingsScreen from './Screens/SettingsScreen';
// const Drawer = createDrawerNavigator();


function HomeScreen({navigation}) {
  return (
    <View style={styles.container}>
      <Text>Home Screen</Text>
      <Button title="Open Drawer" onPress={() => navigation.openDrawer()} />
    </View>
  );
}

function SettingsScreen() {
  return (
    <View style={styles.container}>
      <Text>Settings Screen</Text>
    </View>
  );
}

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
  KeyboardAvoidingView,
  StyleSheet,
  Modal,
} from 'react-native';

import {Colors, Header} from 'react-native/Libraries/NewAppScreen';
const API_KEY = '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';
const configuration = new Configuration({
  apiKey: API_KEY,
});
const openai = new OpenAIApi(configuration);
const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  inputContainer: {
    margin: 20,
    borderWidth: 1,
    padding: 10,
    borderRadius: 10,
    borderColor: 'gray',
    width: 290,
    height: 60,
  },
  keyboardContainer: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  input: {
    color: 'white',
    flex: 1,
    fontSize: 16,
  },
  button: {
    flex: 1,
    backgroundColor: 'orange',
    height: 60,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
    marginRight: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
function App(): JSX.Element {
  let [imageUrl, setImageUrl] = useState<string | null>(null);
  let [chatText, setChatText] = useState<string | null>(null);
  let [conversation, setConversation] = useState<any>({messages: []});
  let [conversations, setConversations] = useState<any>([]);
  let [prompt, setPrompt] = useState<string>('');
  const inputRef = useRef(null);
  const scrollViewRef = useRef();
  const isDarkMode = useColorScheme() === 'dark';
  useEffect(() => {
    // Scroll to the bottom of the ScrollView whenever new messages are added
    scrollViewRef.current.scrollToEnd({animated: true});
  }, [conversation.messages]);

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
    color: 'white',
  };

  function newConversation() {
    setConversation({id: 1, messages: []});
    setPrompt('');
  }
  async function chat() {
    console.log('Chat started');
    const updatedConversation = {
      id: conversation.id,
      messages: [...conversation.messages, {role: 'user', content: prompt}],
    };
    setConversation(updatedConversation);
    console.log(updatedConversation);
    inputRef.current.clear();
    const response = await openai.createChatCompletion({
      messages: updatedConversation.messages,
      model: 'gpt-3.5-turbo',
      max_tokens: 200,
      // stream: true,
    });

    const updatedConversationWithResponse = {
      id: updatedConversation.id,
      messages: [
        ...updatedConversation.messages,
        {role: 'assistant', content: response.data.choices[0].message.content},
      ],
    };
    setConversation(updatedConversationWithResponse);
    console.log(conversation);
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
      setImageUrl(response.data.data[0].url);
    } catch (error) {
      console.error('error: ', error);
    }
  }
  return (
    <SafeAreaView style={{backgroundColor: 'rgb(52, 53, 65)', flex: 1}}>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      {/* <NavigationContainer>
        <Drawer.Navigator initialRouteName="Home">
          <Drawer.Screen name="Home" component={HomeScreen} />
          <Drawer.Screen name="Settings" component={SettingsScreen} />
        </Drawer.Navigator>
      </NavigationContainer> */}
      <View
        style={{
          backgroundColor: isDarkMode ? Colors.black : Colors.white,
          alignItems: 'center',
          flexDirection: 'row', // add this
          justifyContent: 'space-between',
          paddingHorizontal: 30,
          padding: 10,
        }}>
        <TouchableOpacity
          style={{
            borderRadius: 5,
            width: 20,
            height: 20,
            marginLeft: 0,
            alignItems: 'flex-start',
          }}
          onPress={chat}>
          <Text
            style={{
              color: 'white',
              fontSize: 16,
              fontWeight: 'bold',
            }}>
            m
          </Text>
        </TouchableOpacity>

        <Text
          style={{
            fontSize: 20,
            fontWeight: 'bold',
            margin: 10,
            color: isDarkMode ? Colors.white : Colors.black,
          }}>
          GPT-3 Mobile App
        </Text>

        <TouchableOpacity
          style={{
            // backgroundColor: 'blue',
            borderRadius: 5,
            width: 20,
            height: 20,
            marginLeft: 30,
            justifyContent: 'center',
            alignItems: 'flex-end',
          }}
          onPress={newConversation}>
          <Text
            style={{
              color: 'white',
              fontSize: 20,
              fontWeight: 'bold',
            }}>
            +
          </Text>
        </TouchableOpacity>
      </View>
      <ScrollView
        ref={scrollViewRef}
        style={{flex: 1}}
        contentInsetAdjustmentBehavior="automatic">
        {conversation.messages.map((message, index) => (
          <View
            key={index}
            style={{
              padding: 10,
              backgroundColor:
                index % 2 === 0 ? 'rgb(68, 70, 84)' : 'rgb(52, 53, 65)',
            }}>
            <Text
              style={{
                fontSize: 18,
                margin: 10,
                color: isDarkMode ? 'white' : 'white',
              }}>
              {message.content}
            </Text>
          </View>
        ))}
      </ScrollView>
      <KeyboardAvoidingView behavior="padding" style={styles.container}>
        <View style={styles.inputContainer}>
          <TextInput
            ref={inputRef}
            style={styles.input}
            placeholder="Prompt here"
            placeholderTextColor={isDarkMode ? 'white' : 'black'}
            multiline={true}
            onChangeText={newText => setPrompt(newText)}
          />
        </View>
        <TouchableOpacity style={styles.button} onPress={chat}>
          <Text style={styles.buttonText}>Chat</Text>
        </TouchableOpacity>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
export default App;
