import WebSocket from "ws";
import fs from "fs"; // Import file system module

const url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01";

const ws = new WebSocket(url, {
  headers: {
    "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
    "OpenAI-Beta": "realtime=v1",
  },
});

// Handle connection open
ws.on("open", () => {
  console.log("Connected to server.");

  // Send a response creation event to initiate the conversation
  ws.send(JSON.stringify({
    type: "response.create",
    response: {
      modalities: ["text"],
      instructions: "How big is earth?",
    },
  }));
});

// Handle incoming messages from the WebSocket
ws.on("message", (message) => {
  const data = JSON.parse(message.toString());

  // Check for response events and write them to a file
  if (data.response)
    fs.appendFile("responses.log", JSON.stringify(data.response, null, 2) + "\n", (err) => {
    });
});

// Handle WebSocket errors
ws.on("error", (error) => {
  console.error("WebSocket error:", error);
});

// Handle WebSocket close events
ws.on("close", (code, reason) => {
  console.log(`WebSocket closed: ${code} - ${reason}`);
});
