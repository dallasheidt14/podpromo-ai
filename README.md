# PodPromo - AI-Powered Podcast Clip Generation

Transform your podcast episodes into viral social media clips with AI-powered analysis and scoring.

## 🎯 **Plan Structure**

### **Free Plan**
- **2 uploads per month**
- Basic AI scoring
- Standard processing
- Perfect for getting started

### **Pro Plan - $12.99/month**
- **Unlimited uploads**
- Priority processing
- Advanced analytics
- Premium support

## 🚀 **Features**

- **AI-Powered Analysis**: Advanced algorithms detect viral moments
- **Smart Segmentation**: Automatically find the best clip candidates
- **Platform Optimization**: Tailored for TikTok, Reels, YouTube Shorts
- **Genre-Aware Scoring**: Optimized for different podcast types
- **User Management**: Secure authentication with Supabase
- **Payment Processing**: Seamless Paddle integration

## 🛠 **Tech Stack**

### **Backend**
- FastAPI (Python)
- Whisper (Audio transcription)
- Librosa (Audio analysis)
- Supabase (Database & Auth)
- Paddle (Payment processing)

### **Frontend**
- Next.js 14 (React)
- TypeScript
- Tailwind CSS
- Context API for state management

## 📋 **Setup Instructions**

### **1. Backend Setup**

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp env.example .env

# Edit .env with your credentials
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
PADDLE_API_KEY=your_paddle_api_key
PADDLE_WEBHOOK_SECRET=your_webhook_secret
PADDLE_PRO_PRODUCT_ID=your_pro_product_id
APP_URL=http://localhost:3000

# Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Frontend Setup**

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### **3. Supabase Setup**

1. Create a new project at [supabase.com](https://supabase.com)
2. Get your project URL and anon key
3. Create the following tables:

#### **profiles**
```sql
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  email TEXT NOT NULL,
  name TEXT NOT NULL,
  plan TEXT DEFAULT 'free',
  subscription_id TEXT,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### **user_memberships**
```sql
CREATE TABLE user_memberships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  plan TEXT NOT NULL,
  subscription_id TEXT,
  status TEXT DEFAULT 'active',
  start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  end_date TIMESTAMP WITH TIME ZONE
);
```

#### **usage_logs**
```sql
CREATE TABLE usage_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  action TEXT NOT NULL,
  details JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### **4. Paddle Setup**

1. Create account at [paddle.com](https://paddle.com)
2. Create a Pro product:
   - Name: "Pro Plan"
   - Price: $12.99/month
   - Billing: Subscription
   - Get the Product ID
3. Set up webhooks for subscription events
4. Add your API key and webhook secret to `.env`

## 🔧 **API Endpoints**

### **Authentication**
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile

### **Payments**
- `GET /api/paddle/plans` - Get available plans
- `POST /api/paddle/checkout` - Create checkout session
- `POST /api/paddle/webhook` - Handle Paddle webhooks

### **Usage**
- `GET /api/usage/summary/{user_id}` - Get usage summary

### **Core Features**
- `POST /api/upload` - Upload podcast file
- `GET /api/candidates` - Get AI-scored clip candidates
- `POST /api/render-one` - Render single clip

## 🎨 **Frontend Components**

- **AuthContext**: Manages authentication state
- **LoginForm**: User login interface
- **SignupForm**: User registration interface
- **Dashboard**: Main user dashboard
- **UsageDisplay**: Shows usage statistics
- **UpgradeButton**: Handles plan upgrades

## 🔒 **Security Features**

- JWT-based authentication with Supabase
- Webhook signature verification for Paddle
- Rate limiting and usage tracking
- Secure file upload validation

## 📱 **Usage Flow**

1. **User signs up** → Gets free plan (2 uploads/month)
2. **Uploads podcast** → AI analyzes and scores content
3. **Generates clips** → Gets viral moment candidates
4. **Upgrades to Pro** → Unlimited uploads via Paddle
5. **Continues creating** → No more monthly limits

## 🚀 **Deployment**

### **Backend (Production)**
```bash
# Set environment variables
PADDLE_ENVIRONMENT=production
APP_URL=https://yourdomain.com

# Use production server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### **Frontend (Production)**
```bash
npm run build
npm start
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

## 🆘 **Support**

For support and questions:
- Create an issue in this repository
- Check the documentation
- Review the code examples

---

**Built with ❤️ for podcast creators who want to go viral**
