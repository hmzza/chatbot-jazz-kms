function appendMessage(text, sender) {
    let bubble = $('<div>').addClass('chat-bubble').addClass(sender);
  
    if (sender === 'bot') {
      const logo = $('<img>')
        .attr('src', 'static/assets/Jazz-Company-logo-.png')
        .attr('alt', 'Jazz Logo')
        .css({ width: '20px', 'margin-right': '8px' });
  
      const messageContent = $('<span>').text(text);
      bubble.append(logo).append(messageContent);
    } else {
      bubble.text(text);
    }
  
    $('#chat-box').append(bubble);
    $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
    return bubble;
  }
  
  function sendMessage() {
    const input = $('#user-input');
    const userMsg = input.val().trim();
    if (!userMsg) return;
  
    appendMessage(userMsg, 'user');
    input.val('');
  
    // Create bot typing bubble with empty content that will be filled
    const bubble = $('<div>').addClass('chat-bubble').addClass('bot');
    const logo = $('<img>')
      .attr('src', 'static/assets/Jazz-Company-logo-.png')
      .attr('alt', 'Jazz Logo')
      .css({ width: '20px', 'margin-right': '8px' });
    
    const messageContent = $('<span>').text('');
    bubble.append(logo).append(messageContent);
    $('#chat-box').append(bubble);
    $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
  
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMsg })
    })
    .then(response => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Read and process the stream chunks
      function processStream() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            return;
          }
          
          // Decode the current chunk and add it to the message
          const chunk = decoder.decode(value, { stream: true });
          const currentText = messageContent.text();
          messageContent.text(currentText + chunk);
          
          // Auto-scroll
          $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
          
          // Continue reading the stream
          return processStream();
        });
      }
  
      // Start processing the stream
      return processStream().catch(err => {
        messageContent.text("Error: " + err.message);
      });
    })
    .catch(error => {
      bubble.remove();
      appendMessage("Error: " + error.message, 'bot');
    });
  }
  
  function sendQuickReply(message) {
    $('#user-input').val(message);
    sendMessage();
  }
  
  $(document).ready(function () {
    setTimeout(() => {
      appendMessage("Hello! I'm your Jazz Support Bot powered by AI. How can I help you today?", 'bot');
    }, 500);
    
    document.getElementById("user-input").addEventListener("keypress", function (e) {
      if (e.key === "Enter") sendMessage();
    });
  });