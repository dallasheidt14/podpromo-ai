# YouTube Cookies Setup Guide

**Note**: This is an optional advanced setup. Most YouTube videos work without cookies. Only set this up if you encounter "Sign in to confirm you're not a bot" errors.

## When Do You Need This?

- ✅ **Most videos work fine** without any setup
- ⚠️ **Some videos** may show "Sign in to confirm you're not a bot" 
- 🔧 **Advanced users** can set up cookies to handle these cases

## Quick Setup (Optional)

### Option A: Cookie File (Most Reliable)

1. **Install a cookie exporter extension** in your browser:
   - Chrome: "Get cookies.txt" extension
   - Firefox: "cookies.txt" extension
   - Edge: "Get cookies.txt" extension

2. **Export cookies for YouTube**:
   - Go to https://www.youtube.com
   - Make sure you're signed in
   - Click the extension icon
   - Select "Export cookies for this site"
   - Save as `youtube.txt` in a folder like `C:\Users\<yourname>\cookies\`

3. **Set environment variable**:
   ```bash
   set YT_COOKIES_FILE=C:\Users\<yourname>\cookies\youtube.txt
   ```

4. **Restart your backend**:
   ```bash
   uvicorn backend.main:app --reload
   ```

### Option B: Browser Cookie Extraction (Alternative)

1. **Set environment variable**:
   ```bash
   set YT_COOKIES_FROM_BROWSER=chrome:Default
   # or for other browsers:
   # set YT_COOKIES_FROM_BROWSER=edge:Default
   # set YT_COOKIES_FROM_BROWSER=firefox:default
   ```

2. **Restart your backend**:
   ```bash
   uvicorn backend.main:app --reload
   ```

## Verification

After setting up cookies, you should see logs like:
```
INFO:services.youtube_service:Probing YouTube URL with mode=android_with_cookies[file(C:\Users\...\youtube.txt)]
INFO:services.youtube_service:YouTube probe successful [android_with_cookies[file(C:\Users\...\youtube.txt)]]: Video Title (120.00s)
```

## Troubleshooting

### "Sign in to confirm you're not a bot" still appears
- Make sure you're signed into YouTube in your browser
- Try refreshing the cookies file
- Check that the file path is correct and accessible

### "No cookies found" in logs
- Verify the cookie file exists and is readable
- Check the file format (should be Netscape format)
- Try the browser cookie extraction method instead

### Videos still fail after cookies
- Some videos may be region-locked or have other restrictions
- Try different videos to test
- Check if the video is available in your region

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `YT_COOKIES_FILE` | Path to cookies.txt file | `C:\Users\name\cookies\youtube.txt` |
| `YT_COOKIES_FROM_BROWSER` | Browser profile for cookie extraction | `chrome:Default` |
| `YT_PLAYER_CLIENTS` | Player clients to try | `web,android,ios` |

## Security Notes

- Cookie files contain sensitive authentication data
- Keep them secure and don't share them
- Consider using a dedicated YouTube account for this purpose
- Cookies expire periodically and may need refreshing
