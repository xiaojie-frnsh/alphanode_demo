# AlphaNode Demo

AlphaNode is a Django-based backend service that analysts can ask specific questions to customized files.

## 🚀 Features

- Django-based backend with GraphQL API
- Integration with OpenAI and Perplexity AI  
- Vector store implementation
- CORS support for development
- Media file handling
- Secure configuration management

## 🛠️ Installation

1. Clone the repository
```
bash
git clone https://github.com/xiaojie-frnsh/alphanode_demo.git
cd alphanode_demo
```
2. Create and activate a virtual environment
```
bash
python -m venv .venv
source .venv/bin/activate # On Windows, use .venv\Scripts\activate
```
3. Install dependencies
```
bash
pip install -r requirements.txt
```

4. Set up environment variables
Create a `.env` file in the root directory and add:
```
DJANGO_SECRET_KEY=your_secret_key_here
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

5. Run migrations
```
bash
python manage.py migrate
```

6. Start the development server
```
bash
python manage.py runserver
```


## 🔧 Configuration

The project uses Django's settings module for configuration. Key settings include:
- GraphQL endpoint configuration
- Vector store settings
- Media file handling
- CORS settings (configured for development)

## 📚 API Documentation

The API is built using GraphQL. You can explore the API using GraphiQL interface at:
```
http://localhost:8000/graphql/
```


## 🔐 Security

- Sensitive data is managed through environment variables
- CSRF protection enabled
- Django's built-in security features

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Xiaojie - xiaojiezhang1990@outlook.com

Project Link: [https://github.com/xiaojie-frnsh/alphanode_demo](https://github.com/xiaojie-frnsh/alphanode_demo)