// Creates a server that prints new client's IP addresses. 

import processing.net.*;

int port = 65432;   
Server myServer;    

void setup()
{
    size(600, 400);
    background(54);
    myServer = new Server(this, port); // Starts a server on port 10002
}

void draw() {
    background(54); // clear
    
    Client client = myServer.available();
    // If the client is not null, and says something, display what it said
    if (client !=null) {
        String mesg = client.readString();
        if (mesg != null) {
            JSONObject data = parseJSONObject(mesg);
            float x = data.getFloat("x");
            float y = data.getFloat("y");
            //String info = data.getString("info"); 
            int circle_x = int((1 + x) / 2 * width);
            int circle_y = int((1 + y) / 2 * height);
            ellipse(circle_x, circle_y, 33, 33);
        }
    }
}

// ServerEvent message is generated when a new client connects 
// to an existing server.
void serverEvent(Server server, Client client) {
    println("Face detector connected from: " + client.ip());
}
