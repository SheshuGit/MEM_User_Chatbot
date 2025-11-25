"""
Test script for the Wedding Chatbot API
"""
import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
SESSION_ID = f"test_session_{int(time.time())}"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health_check():
    """Test the health check endpoint"""
    print_section("Testing Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat(question):
    """Test the chat endpoint"""
    print_section(f"Testing Chat: '{question}'")
    try:
        payload = {
            "question": question,
            "session_id": SESSION_ID
        }
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 Question: {question}")
            print(f"🤖 Answer: {data.get('answer', 'No answer')}\n")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_categories():
    """Test the categories endpoint"""
    print_section("Testing Categories")
    try:
        response = requests.get(f"{BASE_URL}/api/categories")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            categories = data.get('categories', [])
            print(f"Found {len(categories)} categories:")
            for cat in categories:
                print(f"  - {cat}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_reset():
    """Test the reset endpoint"""
    print_section("Testing Conversation Reset")
    try:
        payload = {"session_id": SESSION_ID}
        response = requests.post(
            f"{BASE_URL}/api/reset",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "🚀"*30)
    print("  WEDDING CHATBOT API TEST SUITE")
    print("🚀"*30)
    
    results = {
        "Health Check": test_health_check(),
        "Categories": test_categories(),
        "Chat - Photographers": test_chat("Show me photographers in Bengaluru"),
        "Chat - Makeup": test_chat("I need makeup artists"),
        "Chat - Price Range": test_chat("Show me vendors within 50000 rupees"),
        "Chat - Follow-up": test_chat("What about their contact information?"),
        "Reset": test_reset(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    print("\n⏳ Starting API tests...")
    print(f"📍 Base URL: {BASE_URL}")
    print(f"🔑 Session ID: {SESSION_ID}")
    
    # Check if server is reachable
    try:
        requests.get(BASE_URL, timeout=2)
    except:
        print("\n❌ ERROR: Cannot connect to the server!")
        print(f"   Make sure the backend is running at {BASE_URL}")
        print("   Run: python app.py")
        exit(1)
    
    # Run tests
    success = run_all_tests()
    
    exit(0 if success else 1)
