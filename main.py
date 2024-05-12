import subprocess
import time

def capture_application_window(app_name, output_file):
    # AppleScript to activate the application and capture its window
    applescript = f'''
        tell application "{app_name}"
            activate
            delay 1  -- Adjust delay as needed
            set frontmost to true
        end tell

        delay 1  -- Adjust delay as needed

        do shell script "screencapture -xW {output_file}"
    '''

    # Execute the AppleScript using osascript
    subprocess.run(["osascript", "-e", applescript])

def capture_entire_screen(output_file):
    time.sleep(5)
    subprocess.run(["screencapture", "-x", output_file])

app_name = "Webex"
output_file = "safari_window.png"
capture_application_window(app_name, output_file)
print(f"Window of {app_name} captured: {output_file}")
