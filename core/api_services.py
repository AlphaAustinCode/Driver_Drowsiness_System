import requests
import threading
from twilio.rest import Client

class APIManager:
    def __init__(self):
        # ── TWILIO CONFIG (For Step 11) ──
        # You will get these from your free Twilio Console
        self.twilio_sid = "AC15cad4afd99d491718eea1580d94c281"
        self.twilio_auth = "a54ae036218915b68e8b6d991fb61f5d"
        self.twilio_from = "+12605253575"  # Your Twilio number
        self.emergency_contact = "+918329754847" # Who gets the text
        
        self.is_fetching_map = False
        self.last_rest_stop = None

    # ─────────────────────────────────────────────
    # STEP 8: OPENSTREETMAP (REST STOPS)
    # ─────────────────────────────────────────────
    
    def fetch_rest_stop_async(self, callback):
        """Spawns a background thread to find a rest stop."""
        if self.is_fetching_map:
            return
            
        self.is_fetching_map = True
        thread = threading.Thread(target=self._query_overpass_api, args=(callback,), daemon=True)
        thread.start()

    def _query_overpass_api(self, callback):
        """Finds the nearest cafe/rest area within 5km of Calangute, Goa."""
        try:
            # Mock GPS Coordinates for Calangute, Goa
            lat, lon = 15.5494, 73.7626 
            
            # Overpass QL query to find cafes within 5000 meters
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            node["amenity"="cafe"](around:5000,{lat},{lon});
            out 1;
            """
            
            response = requests.get(overpass_url, params={'data': query}, timeout=5)
            data = response.json()
            
            if data['elements']:
                name = data['elements'][0].get('tags', {}).get('name', 'Unknown Cafe')
                self.last_rest_stop = name
                print(f"\n[API] Found nearest rest stop: {name}")
                if callback:
                    callback(name)
            else:
                self.last_rest_stop = "Rest Area"
                
        except Exception as e:
            print(f"[API] Map request failed: {e}")
            self.last_rest_stop = "Rest Area"
            
        finally:
            self.is_fetching_map = False

    # ─────────────────────────────────────────────
    # STEP 11: TWILIO (EMERGENCY SMS)
    # ─────────────────────────────────────────────

    def send_emergency_sms_async(self, driver_id, score):
        """Spawns a background thread to send an SOS text."""
        thread = threading.Thread(target=self._send_twilio_sms, args=(driver_id, score), daemon=True)
        thread.start()

    def _send_twilio_sms(self, driver_id, score):
        try:
            if self.twilio_sid == "YOUR_TWILIO_ACCOUNT_SID":
                print("\n[API] Twilio not configured. Simulating Emergency SMS...")
                print(f"[API] Text: SOS! Driver {driver_id} is critically fatigued (Score: {score}). Last known location: Calangute, Goa.")
                return

            client = Client(self.twilio_sid, self.twilio_auth)
            message = client.messages.create(
                body=f"🚨 URGENT: Driver {driver_id} is experiencing CRITICAL FATIGUE (Score: {score}). Intervention required immediately. Location: Calangute, Goa.",
                from_=self.twilio_from,
                to=self.emergency_contact
            )
            print(f"\n[API] Emergency SMS Sent! Message SID: {message.sid}")
            
        except Exception as e:
            print(f"\n[API] Failed to send SMS: {e}")