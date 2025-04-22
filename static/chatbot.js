(function () {
  function appendMessage(text, sender) {
    let bubble = $('<div>').addClass('chat-bubble').addClass(sender);

    if (sender === 'bot') {
      const logo = $('<img>')
        .attr('src', 'static/assets/Jazz-Company-logo-.png')
        .attr('alt', 'Jazz Logo')
        .css({ width: '20px', height: '20px', marginRight: '8px' });
      
      // Process text with proper formatting
      let formattedText = text
        // Bold text (handle ** markdown)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Convert single asterisks at beginning of lines to bullet points
        .replace(/^\s*\*\s+(.+)$/gm, '<li>$1</li>')
        // Handle line breaks
        .replace(/\n/g, '<br>');
      
      // Wrap bullet points in a ul if any exist
      if (formattedText.includes('<li>')) {
        formattedText = formattedText.replace(/<li>(.+?)<\/li>/g, function(match) {
          return match;
        });
        // Find consecutive <li> elements and wrap them in <ul>
        let parts = [];
        let currentText = '';
        const lines = formattedText.split('<br>');
        
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].startsWith('<li>')) {
            if (!currentText.includes('<ul>')) {
              currentText += '<ul>';
            }
            currentText += lines[i];
            if (i === lines.length - 1 || !lines[i+1].startsWith('<li>')) {
              currentText += '</ul>';
            }
          } else {
            if (currentText) {
              parts.push(currentText);
              currentText = '';
            }
            parts.push(lines[i]);
          }
        }
        
        if (currentText) {
          parts.push(currentText);
        }
        
        formattedText = parts.join('<br>');
      }
      
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
    
    // Initialize with an empty message container
    const messageContent = $('<span>').html('');
    bubble.append(logo).append(messageContent);
    $('#chat-box').append(bubble);
    $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);

    // Keep track of accumulated text for formatting
    let accumulatedText = '';

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
            // Final formatting of the complete message
            const finalFormattedText = processText(accumulatedText);
            messageContent.html(finalFormattedText);
            return;
          }
          
          const chunk = decoder.decode(value, { stream: true });
          accumulatedText += chunk;
          
          // For live streaming, apply basic formatting (just bold) as we go
          const partialFormatted = chunk.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
          messageContent.append(partialFormatted);
          
          $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
          
          return processStream();
        });
      }

      // Function to process text with all formatting rules
      function processText(text) {
        // Bold text
        let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Create a temporary div to work with the HTML
        const tempDiv = $('<div>').html(formatted);
        
        // Process the HTML content
        const content = tempDiv.html();
        
        // Handle bullet points (lines starting with *)
        let lines = content.split('\n');
        let inList = false;
        
        for (let i = 0; i < lines.length; i++) {
          const trimmed = lines[i].trim();
          
          if (trimmed.startsWith('* ')) {
            if (!inList) {
              lines[i] = '<ul><li>' + trimmed.substring(2) + '</li>';
              inList = true;
            } else {
              lines[i] = '<li>' + trimmed.substring(2) + '</li>';
            }
            
            // Check if next line is not a bullet point
            if (i === lines.length - 1 || !lines[i+1].trim().startsWith('* ')) {
              lines[i] += '</ul>';
              inList = false;
            }
          } else if (inList) {
            lines[i-1] += '</ul>';
            inList = false;
          }
        }
        
        return lines.join('<br>');
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
    
    // Add click event for the send button if it exists
    $("#send-button").on("click", sendMessage);
  });
})();