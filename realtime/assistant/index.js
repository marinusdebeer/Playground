import Fastify from 'fastify';
import WebSocket from 'ws';
import fs from 'fs';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';

dotenv.config();

// Retrieve the OpenAI API key from environment variables.
const { OPENAI_API_KEY } = process.env;
console.log('OpenAI API Key:', OPENAI_API_KEY);
if (!OPENAI_API_KEY) {
  console.error('Missing OpenAI API key. Please set it in the .env file.');
  process.exit(1);
}

// Initialize Fastify
const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Constants
const SYSTEM_MESSAGE = 'You are taking calls for a business. Please provide the best customer service possible. The business is a cleaning company called Zen Zone Cleaning Services. For booking, make sure to get the following information: Name, Address, Phone Number, Email, Date, Time, and any special instructions. For general inquiries, provide information about the services offered, pricing, and availability. For feedback, thank the customer and ask for a review on Google.';
const VOICE = 'echo';
const PORT = process.env.PORT || 5050;

// List of Event Types to log to the console.
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

// Route for Twilio to handle incoming calls
fastify.all('/incoming-call', async (request, reply) => {
  const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
                          <Response>
                            <Say>Hi, how can I help you today?</Say>
                            <Connect>
                                <Stream url="wss://${request.headers.host}/media-stream" />
                            </Connect>
                            <Pause length="60"/>
                        </Response>`;
  reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for media-stream
fastify.register(async (fastify) => {
  console.log('WebSocket route registered');
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
        const sessionUpdate = {
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

    // Open event for OpenAI WebSocket
    openAiWs.on('open', () => {
      console.log('Connected to the OpenAI Realtime API');
      setTimeout(() => {
        try {
          sendSessionUpdate();
        } catch (error) {
          console.error('Error during initial OpenAI WebSocket communication:', error);
        }
      }, 250);
    });

    // Listen for messages from the OpenAI WebSocket
    openAiWs.on('message', (data) => {
      try {
        const response = JSON.parse(data);

        // Write to file
        if (response.type === 'conversation.item.input_audio_transcription.completed') {
          fs.appendFile('response.json', "\nUser: " + response.transcript, (err) => {
            if (err) console.error('Error appending to response.json:', err);
          });
        }

        if (FILE_EVENTS.includes(response.type)) {
          const assistantTranscript = response?.response?.output?.[0]?.content?.[0]?.transcript;
          if (assistantTranscript) {
            fs.appendFile('response.json', "\nAssistant: " + assistantTranscript, (err) => {
              if (err) console.error('Error appending to response.json:', err);
            });
          }
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
