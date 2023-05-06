/* eslint-disable react-native/no-inline-styles */
import React, { useState, useRef, useEffect } from 'react';
import EventSource from 'react-native-event-source';
import { Configuration, OpenAIApi } from 'openai';
import 'react-native-url-polyfill/auto';
import { API_KEY } from '@env';
// import dotenv from 'dotenv';
// dotenv.config();
import AsyncStorage from '@react-native-async-storage/async-storage';
import ConversationList from './ConversationList';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  useColorScheme,
  View,
  TouchableOpacity,
  Text,
  Image,
  TextInput,
  KeyboardAvoidingView,
  StyleSheet,
  Animated,
} from 'react-native';

import { Colors, Header } from 'react-native/Libraries/NewAppScreen';
const configuration = new Configuration({
  // apiKey: process.env.API_KEY,
  apiKey: API_KEY,
});
const openai = new OpenAIApi(configuration);
const conversationsContainer = {
  zIndex: 1,
  flex: 1,
  position: 'absolute',
  left: 0,
  right: 0,
  backgroundColor: '#222',
  width: '80%',
};
const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  inputContainer: {
    flex: 1,
    margin: 20,
    borderWidth: 1,
    padding: 10,
    borderRadius: 10,
    borderColor: 'gray',
    // width: 290,
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
    // flex: 1,
    backgroundColor: 'orange',
    height: 60,
    width: 70,
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
  let [menu, setMenu] = useState<boolean>(false);
  let [imageUrl, setImageUrl] = useState<string | null>(null);
  let [conversationId, setConversationId] = useState<number>(1);
  let [conversation, setConversation] = useState<number | null>(1);
  let [conversations, setConversations] = useState<any>([
    { id: 1, messages: [], title: 'conversation: 1' },
  ]);
  const slideAnim = useRef(new Animated.Value(-300)).current;
  const promptRef = React.useRef(null);
  let [prompt, setPrompt] = useState<string>('');
  const [title, setTitle] = useState<string>('');
  const scrollViewRef = useRef();
  const isDarkMode = useColorScheme() === 'dark';
  const storeData = async (key, value) => {
    try {
      await AsyncStorage.setItem(key, value);
      console.log('Data successfully saved');
    } catch (error) {
      // Handle error
      console.log('Error saving data: ', error);
    }
  };
  const handleHideConversations = (value) => {
    Animated.timing(slideAnim, {
      toValue: -500,
      duration: 300,
      useNativeDriver: true,
    }).start(() => setMenu(value));
  };

  useEffect(() => {
    if (menu) {
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(slideAnim, {
        toValue: -500,
        duration: 500,
        useNativeDriver: true,
      }).start();
    }
  }, [menu, slideAnim]);

  const getData = async () => {
    try {
      const value = await AsyncStorage.getItem('conversations');
      if (value) {
        setConversations(JSON.parse(value));
        // console.log('Data successfully retrieved');
        // console.log(JSON.parse(value));
        return JSON.parse(value);
      } else {
        storeData('conversations', JSON.stringify(conversations));
      }
    } catch (e) {
      // error reading value
    }
  };
  useEffect(() => {
    (async () => {
      const initialData = await getData();
      if (!initialData || initialData.length === 0) {
        newConversation();
        return;
      }
      // console.log('id: ', initialData[initialData.length - 1].id + 1);
      setConversationId(initialData[initialData.length - 1].id + 1);
      setConversations(initialData);
      setConversation(initialData[0].id);
      setTitle(initialData[0].title);
      // console.log("you're in the useEffect");
      // newConversation();
    })();
  }, []);
  const handleContentSizeChange = () => {
    if (scrollViewRef.current)
      scrollViewRef.current.scrollToEnd({ animated: true });
  };
  async function handleSetConversations(conversations) {
    // console.log(conversations);
    storeData('conversations', JSON.stringify(conversations));
    // await AsyncStorage.setItem('@conversations', JSON.stringify(conversations));
    setConversations(conversations);
  }
  const handleConversationDelete = conversationId => {
    const updatedConversations = conversations.filter(conv => {
      if (conv.id === conversationId) {
        return false;
      }
      return true;
    });

    if (updatedConversations.length === 0) {
      newConversation();
      // handleSetConversations(updatedConversations);
      // setConversation(null);
    } else {
      handleSetConversations(updatedConversations);
      setConversation(updatedConversations[0].id);
    }
    setMenu(true);
  };
  const handleConversationPress = conversationId => {
    const updatedConversations = conversations.filter(conv => {
      if (conv.messages.length === 0 && conv.id !== conversationId) {
        return false;
      }
      return true;
    });
    handleSetConversations(updatedConversations);
    setConversation(conversationId);
    setTitle(conversations.find(el => el.id === conversationId).title);
    // setMenu(false);
    handleHideConversations(false);
  };

  function newConversation() {
    const conv = conversations.find(el => el.id === conversation);
    if (!conv || conv.messages.length > 0) {
      // setMenu(false);
      handleHideConversations(false);
      let newId = conversationId + 1;
      const updatedConversation = {
        id: newId,
        messages: [],
        title: 'New Chat',
      };
      setTitle('New Chat');
      setConversation(conversationId + 1);
      setPrompt('');
      handleSetConversations([...conversations, updatedConversation]);
      // setConversations([...conversations, updatedConversation]);
      // console.log(conversation);
      setConversationId(conversationId + 1);
    } else {
      // setConversation(conversationId);
      // setMenu(false);
      handleHideConversations(false);
    }
  }

  function updateConversation(current, user: string, msg: string) {
    const updatedConversation = {
      ...current,
      messages: [...current.messages, { role: user, content: msg }],
    };
    const updatedConversations = conversations.map(el => {
      if (el.id === conversation) {
        return updatedConversation;
      }
      return el;
    });
    handleSetConversations(updatedConversations);
    // setConversations(updatedConversations);
    // console.log(updatedConversations);
    return updatedConversation;
  }
  const calculateFontSize = (text) => {
    const baseFontSize = 20;
    const maxLength = 23; // Adjust this value based on the desired maximum length before scaling the font size
    if (text.length > maxLength) {
      const fontSize = baseFontSize * (maxLength / text.length);
      return fontSize;
    }
    return baseFontSize;
  };
  async function openai_chat(msgs) {
    const response = await openai.createChatCompletion({
      messages: msgs,
      // model: 'gpt-4',
      model: 'gpt-3.5-turbo',
      // max_tokens: 200,
      // stream: true,
    });
    return response.data.choices[0].message.content;
  }
  async function chat() {
    if (!prompt || prompt.length === 0) {
      return;
    }
    console.log('Chat started');
    let current = conversations.find(el => el.id === conversation);
    current = updateConversation(current, 'user', prompt);
    promptRef.current.clear();
    // console.log(current.messages);
    let msg = await openai_chat(current.messages);
    current = updateConversation(current, 'assistant', msg);
    if (current.messages.length === 2) {
      const titleMsg = {
        role: 'user',
        content: 'create a short 4 word title based on the following: ' + prompt,
      };
      current.title = await openai_chat([titleMsg]);
      const conv = conversations.map(el => {
        if (el.id === conversation) {
          setTitle(current.title);
          return current;
        }
        return el;
      });
      handleSetConversations(conv);
    }
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
    <SafeAreaView
      style={{ backgroundColor: 'rgb(52, 53, 65)', flex: 1, height: '100%' }}>
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
            marginLeft: 0,
            alignItems: 'flex-start',
          }}
          onPress={() =>
            !menu ? setMenu(true) : handleHideConversations(false)
          }>
          <Text
            style={{
              color: isDarkMode ? 'white' : 'black',
              fontSize: 16,
              padding: 10,
              fontWeight: 'bold',
            }}>
            m
          </Text>
        </TouchableOpacity>

        <Text
          numberOfLines={1}
          style={{
            fontSize: calculateFontSize(title),
            fontWeight: 'bold',
            margin: 10,
            maxWidth: '70%',
            color: isDarkMode ? Colors.white : Colors.black,
          }}>
          {title}
          {/* {menu ? 'GPT' : title} */}
        </Text>
        <TouchableOpacity
          style={{
            // backgroundColor: 'blue',
            borderRadius: 5,
            marginLeft: 30,
            justifyContent: 'center',
            alignItems: 'flex-end',
          }}
          onPress={newConversation}>
          <Text
            style={{
              color: isDarkMode ? 'white' : 'black',
              fontSize: 20,
              padding: 10,
              fontWeight: 'bold',
            }}>
            +
          </Text>
        </TouchableOpacity>
      </View>

      <View style={{ flex: 1, backgroundColor: 'rgba(68, 70, 84, 0.6)'}}>
        <ScrollView
          ref={scrollViewRef}
          style={{ flex: 1, zIndex: 0 }}
          onContentSizeChange={handleContentSizeChange}
          contentInsetAdjustmentBehavior="automatic">
          {conversations
            ?.find(el => el.id === conversation)
            ?.messages?.map((message, index) => (
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
          {/* <View style={{height: 500}}></View> */}
        </ScrollView>
        <Animated.View
          style={{
            ...conversationsContainer,
            transform: [{translateX: slideAnim}],
          }}>
          <ConversationList
            conversations={conversations}
            onConversationPress={handleConversationPress}
            onConversationDelete={handleConversationDelete}
          />
        </Animated.View>
      </View>

      {!menu && conversations.length > 0 ? (
        <KeyboardAvoidingView behavior="padding" style={styles.container}>
          <View style={styles.inputContainer}>
            <TextInput
              ref={promptRef}
              style={styles.input}
              placeholder="Prompt here"
              placeholderTextColor={'white'}
              multiline={true}
              onChangeText={newText => setPrompt(newText)}
            />
          </View>
          <TouchableOpacity style={styles.button} onPress={chat}>
            <Text style={styles.buttonText}>Chat</Text>
          </TouchableOpacity>
        </KeyboardAvoidingView>
      ) : (
        ''
      )}
    </SafeAreaView>
  );
}

export default App;
