# PC-Service discovery announce — why nobody upstream types IPs (and we did)

The Quest app auto-discovers the PC service on the LAN. `UIUdpReceiver.cs` listens on
**UDP :29888** from app start; on a valid announce it pops an **IP-select dialog** — one
button per advertised IP — and a single trigger-click runs the full Network-panel connect
(`TcpConnect(ip)` → TCP :63901). Select-from-list, not zero-click, but no keyboard.
(Upstream client README documents this as the intended UX.)

The real PC service broadcasts the announce every **5 s** to the **/24 subnet broadcast**
`x.y.z.255:29888` (`tcpserverworker.cpp::onBraodCastTCPServer`, constants in
`Manage_global.h`).

**Why we were typing IPs:** our PC service runs in Docker (`xr-bridge`) on macOS. Inside the
container it enumerates *container* interfaces → the broadcast stays on the Docker bridge
subnet AND would advertise the container IP. Doubly broken. Fix: `scripts/xrtk_announce.py`
replays the announce natively on the Mac, advertising the Mac's LAN IP (whose published
:63901 reaches the container).

## Wire format

`#pragma pack(1)`, multi-byte fields little-endian; total `15 + len(ip)` bytes:

| offset | size | value |
|---|---|---|
| 0 | 1 | `0xCF` head (`TCP_SERVER_HEAD_CODE`) |
| 1 | 1 | `0x7E` cmd (`UDP_PACKET_CMD_TCPIP`) |
| 2 | 4 | uint32 LE — length of IP string |
| 6 | N | ASCII IP string, e.g. `"192.168.0.190"` |
| 6+N | 8 | uint64 LE — epoch seconds (client parses, ignores) |
| 14+N | 1 | `0xA5` tail (NOT validated on the UDP path) |

Client accept rules (`PackageHandle.Unpack(byte[])` + `NetCMD.cs`): `data[0]==0xCF`,
`data[1]==0x7E`, length at `[2:6]`, body at `[6:6+len)`, total length `> 13+len`.

## Related findings (same investigation, 2026-06-11)

- **Network panel IP is NEVER persisted** (`IpInputDialog.cs` — no PlayerPrefs). By design;
  discovery is the intended path.
- **Camera/ZEDMINI Listen IP IS persisted** (PlayerPrefs key `"CameraSendInputDialog"`) —
  type it once, it pre-fills forever after. Caveat: `PlayerPrefs.Save()` is never called
  explicitly, so force-killing the app right after the first entry can lose it; exit clean.
- Video-source dropdown is rebuilt from YAML each launch; ZEDMINI is first entry → default.
- No intent extras / deep links / config file for IPs (stock UnityPlayerActivity manifest).
  `video_source.yml` IS overridable at the app's `persistentDataPath` but holds no IPs.
- If the announce doesn't arrive (some Android stacks filter subnet broadcasts; no
  MulticastLock taken), unicast the same packet straight to the Quest's IP — the client
  binds `0.0.0.0:29888`, so unicast is always delivered. `xrtk_announce.py --unicast <ip>`.
