chrome.webRequest.onBeforeRequest.addListener(
function(info) {
return {redirectUrl: "https://raw.github.com/Thiece/bilibili-safe/master/tv.bilibili.player.xml"};
},
{
urls: [
"http://interface.bilibili.tv/blocklist"
]
},
["blocking"]
);
