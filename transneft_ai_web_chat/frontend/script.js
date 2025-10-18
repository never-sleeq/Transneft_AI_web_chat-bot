document.addEventListener('DOMContentLoaded', function() {
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const messagesContainer = document.getElementById('messages-container');
    const avatarImage = document.getElementById('avatar-image');

    let animationTimer = null;

    function setAvatarAnimation(animationName) {
        if (animationTimer) {
            clearTimeout(animationTimer);
            animationTimer = null;
        }

        avatarImage.src = `assets/${animationName}.gif`;
        console.log(`Аватар переключен на: ${animationName}`);

        if (animationName !== 'idle') {
            let timeout = 3000;
            
            animationTimer = setTimeout(() => {
                setAvatarAnimation('idle');
            }, timeout);
        }
    }

    function addMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const time = new Date().toLocaleTimeString('ru-RU', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        messageDiv.innerHTML = `
            <div class="message-text">${text}</div>
            <div class="message-time">${time}</div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }


    function openChatAnimation() {
        setAvatarAnimation('wave');
    }

    function engagementAnimation() {
        setAvatarAnimation('suggest');
    }

    function initChat() {
        openChatAnimation();
        
        setTimeout(() => {
            addMessage('Здравствуйте! Я ваш цифровой помощник компании "Транснефть". Чем могу помочь?', false);
            
            setTimeout(() => {
                engagementAnimation();
            }, 1000);
            
        }, 1000);
    }

    setTimeout(initChat, 500);

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        addMessage(text, true);
        userInput.value = '';
        setAvatarAnimation('idle');

        try {
            const response = await fetch('http://localhost:8001/api/qa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, top_k: 3 })
            });

            if (!response.ok) {
                const err = await response.text();
                throw new Error(`Ошибка ${response.status}: ${err}`);
            }

            const data = await response.json();
            addMessage(data.answer, false);
        } catch (error) {
            console.error("Ошибка:", error);
            addMessage("❌ Не удалось получить ответ от ИИ. Проверь консоль.", false);
        }

        setTimeout(() => engagementAnimation(), 1000);
    }
});