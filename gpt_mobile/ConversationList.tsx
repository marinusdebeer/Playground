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
  const renderConversation = ({item}) => (
    <View style={styles.conversationContainer}>
      <TouchableOpacity onPress={() => onConversationPress(item.id)} style={styles.conversation}>
        <View style={styles.conversationItem}>
          <Text style={styles.conversationTitle}>{item.title}</Text>
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
    padding: 10,
    borderRadius: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
    // backgroundColor: '#f5f5f5',
  },
  deleteButton: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'red',
    borderRadius: 4,
    paddingVertical: 5,
    paddingHorizontal: 10,
    marginLeft: 10,
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
    backgroundColor: 'rgb(52, 53, 65)',
  },
  conversationItem: {
    backgroundColor: 'rgb(52, 53, 65)',
    padding: 20,
    width: '100%',
  },
  conversationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  otherComponent: {
    height: 100,
    backgroundColor: '#ccc',
  },
});

export default ConversationList;
