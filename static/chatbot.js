(function () {
  function appendMessage(text, sender) {
    let bubble = $('<div>').addClass('chat-bubble').addClass(sender);

    if (sender === 'bot') {
      const logo = $('<img>')
        .attr('src', 'static/assets/Jazz-Company-logo-.png')
        .attr('alt', 'Jazz Logo')
        .css({ width: '20px', height: '20px', marginRight: '8px' });
      
      // Format the text: bold and bullet points
      let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\* /g, '<br>• ')  // Convert "newline + * " to "<br>• "
        .replace(/^\* /g, '• ')       // If the text starts with "* "
        .replace(/\n/g, '<br>');      // Replace remaining newlines
      
      const messageContent = $('<span>').html(formattedText);
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

    const bubble = $('<div>').addClass('chat-bubble').addClass('bot');
    const logo = $('<img>')
      .attr('src', 'static/assets/Jazz-Company-logo-.png')
      .attr('alt', 'Jazz Logo')
      .css({ width: '20px', height: '20px', marginRight: '8px' });
    
    const messageContent = $('<span>');
    bubble.append(logo).append(messageContent);
    $('#chat-box').append(bubble);
    $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);

    // Store the complete response
    let fullResponse = '';

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMsg })
    })
    .then(response => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      function processStream() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            // Apply final formatting for the complete message
            let formattedText = fullResponse
              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/\n\* /g, '<br>• ')  // Convert "newline + * " to "<br>• "
              .replace(/^\* /g, '• ')       // If the text starts with "* "
              .replace(/\n/g, '<br>');      // Replace remaining newlines
            
            messageContent.html(formattedText);
            return;
          }
          
          const chunk = decoder.decode(value, { stream: true });
          fullResponse += chunk;
          
          // During streaming, show with simple formatting
          // (new lines will work, but we'll save full formatting for the end)
          let streamingText = fullResponse.replace(/\n/g, '<br>');
          messageContent.html(streamingText);
          
          $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
          
          return processStream();
        });
      }

      return processStream().catch(err => {
        messageContent.html("Error: " + err.message);
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
      appendMessage(
        "Hello! I'm your Jazz Support Bot powered by AI. How can I help you today?",
        'bot'
      );
    }, 500);

    document.getElementById("user-input").addEventListener("keypress", function (e) {
      if (e.key === "Enter") sendMessage();
    });
  });
})();