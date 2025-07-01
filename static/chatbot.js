(function () {
  let currentController = null; // Store the AbortController

  function appendMessage(text, sender) {
    let bubble = $("<div>").addClass("chat-bubble").addClass(sender);

    if (sender === "bot") {
      const logo = $("<img>")
        .attr("src", "static/assets/Jazz-Company-logo-.png")
        .attr("alt", "Jazz Logo")
        .css({ width: "20px", height: "20px", marginRight: "8px" });

      // Restore original formatting logic
      let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // Bold text
        .replace(/\n/g, "<br>"); // New lines to <br>

      const messageContent = $("<span>").html(formattedText);
      bubble.append(logo).append(messageContent);
    } else {
      bubble.text(text);
    }

    $("#chat-box").append(bubble);
    $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
    return bubble;
  }

  function startTypingAnimation() {
    $("#robotArea").addClass("robot-thinking");
  }

  function stopTypingAnimation() {
    $("#robotArea").removeClass("robot-thinking");
  }

  function disableUI() {
    $("#send-btn").prop("disabled", true);
    $("#user-input").prop("disabled", true);
    $("#quick-replies button").prop("disabled", true);
    $("#stop-btn").show();
    startTypingAnimation();
  }

  function enableUI() {
    $("#send-btn").prop("disabled", false);
    $("#user-input").prop("disabled", false);
    $("#quick-replies button").prop("disabled", false);
    $("#stop-btn").hide();
    $("#user-input").focus();
    stopTypingAnimation();
  }

  function sendMessage() {
    const input = $("#user-input");
    const userMsg = input.val().trim();
    if (!userMsg) return;

    appendMessage(userMsg, "user");
    input.val("");

    // Disable all UI elements
    disableUI();

    const bubble = $("<div>").addClass("chat-bubble").addClass("bot");
    const logo = $("<img>")
      .attr("src", "static/assets/Jazz-Company-logo-.png")
      .attr("alt", "Jazz Logo")
      .css({ width: "20px", height: "20px", marginRight: "8px" });

    const messageContent = $("<span>");
    bubble.append(logo).append(messageContent);
    $("#chat-box").append(bubble);
    $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);

    // Store the complete response
    let fullResponse = "";

    // Create a new AbortController
    currentController = new AbortController();
    const signal = currentController.signal;

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMsg }),
      signal: signal, // Add the signal to allow aborting
    })
      .then((response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        function processStream() {
          return reader.read().then(({ done, value }) => {
            if (done) {
              // Apply final formatting for the complete message
              let formattedText = fullResponse
                .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                .replace(/\n/g, "<br>");

              messageContent.html(formattedText);
              enableUI();
              return;
            }

            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;

            // During streaming, show with simple formatting
            let streamingText = fullResponse.replace(/\n/g, "<br>");
            messageContent.html(streamingText);

            $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);

            return processStream();
          });
        }

        return processStream().catch((err) => {
          if (err.name === "AbortError") {
            messageContent.html(
              fullResponse.replace(/\n/g, "<br>") +
                "<br><em>[Response stopped]</em>"
            );
          } else {
            messageContent.html("Error: " + err.message);
          }
          enableUI();
        });
      })
      .catch((error) => {
        if (error.name === "AbortError") {
          // Do nothing special, already handled
        } else {
          bubble.remove();
          appendMessage("Error: " + error.message, "bot");
        }
        enableUI();
      });
  }

  function stopResponse() {
    if (currentController) {
      currentController.abort();
      currentController = null;
    }
  }

  function sendQuickReply(message) {
    // Check if UI is currently disabled (response in progress)
    if ($("#quick-replies button").prop("disabled")) {
      return; // Don't send if disabled
    }
    
    $("#user-input").val(message);
    sendMessage();
  }

  // Expose these functions globally
  window.sendMessage = sendMessage;
  window.stopResponse = stopResponse;
  window.sendQuickReply = sendQuickReply;

  $(document).ready(function () {
    // Hide stop button initially
    $("#stop-btn").hide();

    setTimeout(() => {
      appendMessage(
        "Hello! I'm your Jazz Support Bot powered by AI. How can I help you today?",
        "bot"
      );
    }, 500);

    document
      .getElementById("user-input")
      .addEventListener("keypress", function (e) {
        if (e.key === "Enter" && !$("#user-input").prop("disabled")) {
          sendMessage();
        }
      });
  });
})();