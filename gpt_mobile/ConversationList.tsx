import React from 'react';
import {
  View,
  ScrollView,
  FlatList,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';

const ConversationList = ({
  conversations,
  onConversationPress,
  onConversationDelete,
}) => {
  const calculateFontSize = (text) => {
    const baseFontSize = 18;
    const maxLength = 18; // Adjust this value based on the desired maximum length before scaling the font size
    if (text.length > maxLength) {
      const fontSize = baseFontSize * (maxLength / text.length);
      return fontSize;
    }
    return baseFontSize;
  };
  const renderConversation = ({item}) => (
    <View style={styles.conversationContainer}>
      <TouchableOpacity onPress={() => onConversationPress(item.id)} style={styles.conversation}>
        <View style={styles.conversationItem}>
          <Text
            style={{
              ...styles.conversationTitle,
              fontSize: calculateFontSize(item.title),
            }}>
            {item.title}
          </Text>
        </View>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => onConversationDelete(item.id)} style={styles.deleteButton}>
        <Text style={styles.deleteButtonText}>Delete</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        style={{height: '70%'}}
        data={conversations}
        renderItem={renderConversation}
        keyExtractor={item => item.id}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  conversationContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
    paddingVertical: 10,
    borderRadius: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
    // backgroundColor: 'rgba(86,88,105, 1)',
    // backgroundColor: '#f5f5f5',
  },
  deleteButton: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'red',
    borderRadius: 4,
    paddingVertical: 5,
    paddingHorizontal: 10,
    // marginLeft: 10,
    marginRight: 10,
    width: 70,
  },
  deleteButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  conversation: {
    flex: 1,
    // width: '90%',
  },
  container: {
    height: '90%',
    width: '100%',
    // backgroundColor: 'rgba(86,88,105, 1)',
  },
  conversationItem: {
    // backgroundColor: 'rgb(52, 53, 65)',
    padding: 20,
    width: '100%',
  },
  conversationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
});

export default ConversationList;
