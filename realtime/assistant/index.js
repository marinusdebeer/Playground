import Fastify from 'fastify';
import WebSocket from 'ws';
import fs from 'fs';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import twilio from 'twilio';
import OpenAI from "openai";

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Retrieve the OpenAI API key from environment variables. You must have OpenAI Realtime API access.
const { OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER } = process.env;
if (!OPENAI_API_KEY || !TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
  console.error('Missing required API keys or tokens. Please set them in the .env file.');
  process.exit(1);
}

const twilioClient = new twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);

// Initialize Fastify
const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Constants
const SYSTEM_MESSAGE = 'You are taking calls for a business. Please provide the best customer service possible. The business is a cleaning company called Zen Zone Cleaning Services. For booking make sure to get the following information: Name, Address, Phone Number, Email, Date, Time, and any special instructions. For general inquiries, provide information about the services offered, pricing, and availability. For feedback, thank the customer and ask for a review on Google.';
const VOICE = 'echo';
const PORT = process.env.PORT || 5050; // Allow dynamic port assignment

// List of Event Types to log to the console. See OpenAI Realtime API Documentation. (session.updated is handled separately.)
const LOG_EVENT_TYPES = [
  'response.content.done',
  'response.done',
  'session.created',
  'conversation.item.input_audio_transcription.completed',
];

const FILE_EVENTS = [
  'response.content.done',
  'response.done',
  'conversation.item.input_audio_transcription.completed',
];

// Root Route
fastify.get('/', async (request, reply) => {
  reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Route for Twilio to handle incoming and outgoing calls
fastify.all('/incoming-call', async (request, reply) => {
  console.log(request.headers.host);
  const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
                        <Response>
                            <Say>Hi, you are connected to Zen Zone Cleaning Services. How can I help you today?</Say>
                            <Connect>
                                <Stream url="wss://${request.headers.host}/media-stream" />
                            </Connect>
                        </Response>`;
  reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for media-stream
fastify.register(async (fastify) => {
  fastify.get('/media-stream', { websocket: true }, (connection, req) => {
    console.log('Client connected');
    fs.appendFile('response.json', "\nNew call", (err) => {
      if (err) console.error('Error appending to response.json:', err);
    });

    const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "OpenAI-Beta": "realtime=v1"
      }
    });

    let streamSid = null;

    const sendSessionUpdate = () => {
      try {
        let sessionUpdate = {
          type: 'session.update',
          session: {
            turn_detection: { type: 'server_vad' },
            input_audio_format: 'g711_ulaw',
            output_audio_format: 'g711_ulaw',
            voice: VOICE,
            instructions: SYSTEM_MESSAGE,
            modalities: ["text", "audio"],
            temperature: 0.8,
            input_audio_transcription: {
              model: "whisper-1"
            },
          }
        };
        console.log('Sending session update:', JSON.stringify(sessionUpdate));
        openAiWs.send(JSON.stringify(sessionUpdate));
      } catch (error) {
        console.error('Error sending session update to OpenAI:', error);
      }
    };

    // Function to send a message when the call is picked up
    const sendCallPickedUpMessage = () => {
      try {
        const callPickedUpMessage = {
          type: 'conversation.item.create',
          item: {
            role: 'user',
            content: [{
              type: 'input_text',
              text: 'How big is Earth?'
            }]
          }
        };
        console.log('Sending call picked up message:', JSON.stringify(callPickedUpMessage));
        openAiWs.send(JSON.stringify(callPickedUpMessage));
      } catch (error) {
        console.error('Error sending call picked up message to OpenAI:', error);
      }
    };

    // Open event for OpenAI WebSocket
    openAiWs.on('open', () => {
      console.log('Connected to the OpenAI Realtime API');
      setTimeout(() => {
        try {
          sendSessionUpdate();
          sendCallPickedUpMessage(); // Send the message after session update
        } catch (error) {
          console.error('Error during initial OpenAI WebSocket communication:', error);
        }
      }, 250); // Ensure connection stability, send after .25 seconds
    });

    // Listen for messages from the OpenAI WebSocket (and send to Twilio if necessary)
    openAiWs.on('message', (data) => {
      try {
        const response = JSON.parse(data);

        // Write to file
        if (response.type === 'conversation.item.input_audio_transcription.completed') {
          fs.appendFile('response.json', "\nUser: " + response?.transcript, (err) => {
            if (err) console.error('Error appending to response.json:', err);
          });
        }

        if (FILE_EVENTS.includes(response.type)) {
          fs.appendFile('response.json', "\nAssistant: " + response?.response?.output[0]?.content[0]?.transcript, (err) => {
            if (err) console.error('Error appending to response.json:', err);
          });
        }

        if (LOG_EVENT_TYPES.includes(response.type)) {
          console.log(`Received event: ${response.type}`, response);
        }

        if (response.type === 'session.updated') {
          console.log('Session updated successfully:', response);
        }

        if (response.type === 'response.audio.delta' && response.delta) {
          const audioDelta = {
            event: 'media',
            streamSid: streamSid,
            media: { payload: Buffer.from(response.delta, 'base64').toString('base64') }
          };
          try {
            connection.send(JSON.stringify(audioDelta));
          } catch (error) {
            console.error('Error sending audio delta to Twilio:', error);
          }
        }
      } catch (error) {
        console.error('Error processing OpenAI message:', error, 'Raw message:', data);
      }
    });

    // Handle errors on the OpenAI WebSocket
    openAiWs.on('error', (error) => {
      console.error('Error in the OpenAI WebSocket:', error);
      fs.appendFile('response.json', `\nError in the OpenAI WebSocket: ${error.message}`, (err) => {
        if (err) console.error('Error appending to response.json:', err);
      });
    });

    // Handle OpenAI WebSocket close event
    openAiWs.on('close', (code, reason) => {
      console.log(`OpenAI WebSocket closed. Code: ${code}, Reason: ${reason}`);
      fs.appendFile('response.json', `\nDisconnected from the OpenAI Realtime API. Code: ${code}, Reason: ${reason}`, (err) => {
        if (err) console.error('Error appending to response.json:', err);
      });
    });

    // Handle incoming messages from Twilio
    connection.on('message', (message) => {
      try {
        const data = JSON.parse(message);
        switch (data.event) {
          case 'media':
            if (openAiWs.readyState === WebSocket.OPEN) {
              try {
                const audioAppend = {
                  type: 'input_audio_buffer.append',
                  audio: data.media.payload
                };
                openAiWs.send(JSON.stringify(audioAppend));
              } catch (error) {
                console.error('Error sending audio to OpenAI:', error);
              }
            } else {
              console.warn('OpenAI WebSocket is not open. Cannot send audio.');
            }
            break;
          case 'start':
            streamSid = data.start.streamSid;
            console.log('Incoming stream has started', streamSid);
            break;
          default:
            console.log('Received non-media event:', data.event);
            break;
        }
      } catch (error) {
        console.error('Error parsing message from Twilio:', error, 'Message:', message);
      }
    });

    // Handle errors on the Twilio WebSocket connection
    connection.on('error', (error) => {
      console.error('Error in Twilio WebSocket connection:', error);
    });

    // Handle connection close
    connection.on('close', () => {
      if (openAiWs.readyState === WebSocket.OPEN || openAiWs.readyState === WebSocket.CONNECTING) {
        openAiWs.close();
      }
      console.log('Client disconnected.');
      fs.appendFile('response.json', "\nCall Ended", (err) => {
        if (err) console.error('Error appending to response.json:', err);
      });
    });
  });
});

fastify.listen({ port: PORT, host: '0.0.0.0' }, (err) => {
  if (err) {
    console.error('Error starting Fastify server:', err);
    process.exit(1);
  }
  console.log(`Server running on port ${PORT}`);
});
