"""
test_apis.py
============
Standalone test to verify OpenStreetMap and Twilio SMS integrations.
"""

import time
from core.api_services import APIManager

def map_callback(destination_name):
    print(f"\n[OVERLAY UPDATE] Drawing to screen: 'Navigate to {destination_name}'")

if __name__ == "__main__":
    print("Initializing API Manager...")
    api = APIManager()

    print("\n1. Simulating Level 2 Fatigue -> Fetching Rest Stop...")
    api.fetch_rest_stop_async(callback=map_callback)

    print("2. Simulating Level 3 Fatigue -> Sending Emergency SMS...")
    api.send_emergency_sms_async(driver_id=1, score="CRITICAL (0.89)")

    print("\nWaiting for background threads to complete...")
    
    # Keep the script alive for 5 seconds so the background threads can finish
    for i in range(5, 0, -1):
        print(f" ... {i}")
        time.sleep(1)
        
    print("\n[TEST COMPLETE]")