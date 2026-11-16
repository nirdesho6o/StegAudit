"""Generate docs/hero.svg: a 9 s looping flat-cartoon story (CSS keyframes only, no JS, no external resources).

Default (unanimated) styles are the FINAL frame, so prefers-reduced-motion: reduce simply switches animations off.
Usage: hero_gen.py OUT.svg [freeze_seconds]   (freeze_seconds: pause every animation at that time, for checking)
"""
import sys

OUT = sys.argv[1]
FREEZE = float(sys.argv[2]) if len(sys.argv) > 2 else None

INK = "#334155"
BLUE, BLUE_L = "#2563EB", "#DBEAFE"
GREEN, GREEN_L = "#16A34A", "#DCFCE7"
PURPLE, PURPLE_L = "#9333EA", "#F3E8FF"
ORANGE, ORANGE_L = "#EA580C", "#FFEDD5"
RED = "#DC2626"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
TOP = 128          # top of every robot head


def robot(x, col, light, extra=""):
    t = TOP
    return f"""
  <g stroke="{INK}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
    <line x1="{x}" y1="{t-14}" x2="{x}" y2="{t}"/>
    <circle cx="{x}" cy="{t-18}" r="6" fill="{col}"/>
    <rect x="{x-20}" y="{t+104}" width="14" height="20" rx="6" fill="{col}"/>
    <rect x="{x+6}" y="{t+104}" width="14" height="20" rx="6" fill="{col}"/>
    <rect x="{x-30}" y="{t+56}" width="60" height="52" rx="14" fill="{col}"/>
    <rect x="{x-34}" y="{t}" width="68" height="52" rx="16" fill="{light}"/>
    <circle cx="{x-13}" cy="{t+24}" r="5" fill="{INK}" stroke="none"/>
    <circle cx="{x+13}" cy="{t+24}" r="5" fill="{INK}" stroke="none"/>
    <path d="M{x-9} {t+37} Q{x} {t+44} {x+9} {t+37}" fill="none" stroke-width="3"/>
    {extra}
  </g>"""


def label(x, text, col):
    return (f'<text x="{x}" y="{TOP+152}" text-anchor="middle" font-size="18" font-weight="700" '
            f'fill="{col}">{text}</text>')


def key(cx, cy, cls):
    return f"""
  <g class="{cls} pop" stroke="{INK}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="{cx-9}" cy="{cy}" r="8" fill="#FACC15"/>
    <circle cx="{cx-9}" cy="{cy}" r="2.5" fill="{INK}" stroke="none"/>
    <path d="M{cx-1} {cy} H{cx+16} M{cx+10} {cy} V{cy+6} M{cx+16} {cy} V{cy+5}" fill="none"/>
  </g>"""


BX, PX, GX, AX = 105, 300, 520, 850       # Blue, Purple, Green, Auditor centres

css = f"""
  text {{ font-family: {FONT}; }}
  .pop {{ transform-box: fill-box; transform-origin: center; }}
  .a {{ animation-duration: 9s; animation-iteration-count: infinite; animation-fill-mode: both;
        animation-timing-function: ease-in-out; }}
  .key1 {{ animation-name: key; }}
  .key2 {{ animation-name: key; animation-delay: .25s; }}
  .card {{ opacity: 0; animation-name: card; }}
  .note {{ transform: translate(210px, 0); animation-name: note; }}
  .seen {{ animation-name: seen; }}
  .lens {{ animation-name: lens; transform-box: fill-box; transform-origin: 20% 80%; }}
  .glow {{ opacity: 0; animation-name: glow; }}
  .think {{ animation-name: think; }}
  .ans {{ animation-name: ans; }}
  .flag {{ animation-name: flag; }}
  @keyframes key   {{ 0% {{opacity:0; transform:scale(.2)}} 5% {{opacity:1; transform:scale(1.25)}} 8% {{transform:scale(1)}}
                     95% {{opacity:1}} 100% {{opacity:0}} }}
  @keyframes card  {{ 0%,9% {{opacity:0; transform:scaleX(1)}} 12% {{opacity:1}} 21% {{opacity:1; transform:scaleX(1)}}
                     26% {{opacity:1; transform:scaleX(.05)}} 27%,100% {{opacity:0; transform:scaleX(.05)}} }}
  @keyframes note  {{ 0%,25% {{opacity:0; transform:translate(0,0)}} 28%,31% {{opacity:1; transform:translate(0,0)}}
                     47% {{transform:translate(210px,0)}} 95% {{opacity:1; transform:translate(210px,0)}}
                     100% {{opacity:0; transform:translate(210px,0)}} }}
  @keyframes lens  {{ 0%,30% {{transform:rotate(0)}} 34% {{transform:rotate(-12deg)}} 38% {{transform:rotate(8deg)}}
                     42%,100% {{transform:rotate(0)}} }}
  @keyframes seen  {{ 0%,36% {{opacity:0}} 39%,95% {{opacity:1}} 100% {{opacity:0}} }}
  @keyframes glow  {{ 0%,49% {{opacity:0}} 52%,60% {{opacity:1}} 63%,100% {{opacity:0}} }}
  @keyframes think {{ 0%,55% {{opacity:0}} 59%,95% {{opacity:1}} 100% {{opacity:0}} }}
  @keyframes ans   {{ 0%,65% {{opacity:0}} 69%,95% {{opacity:1}} 100% {{opacity:0}} }}
  @keyframes flag  {{ 0%,76% {{opacity:0; transform:scale(.2)}} 80% {{opacity:1; transform:scale(1.2)}} 83%,95% {{opacity:1; transform:scale(1)}}
                     100% {{opacity:0}} }}
  @media (prefers-reduced-motion: reduce) {{ .a {{ animation: none !important; }} }}
"""
if FREEZE is not None:
    css += f"\n  .a {{ animation-play-state: paused !important; animation-delay: -{FREEZE}s !important; }}\n"

# note: drawn at its START position (in front of Blue); translated 160 px to Green
NX, NY = 150, 40
note = f"""
  <g class="note a">
    <rect x="{NX}" y="{NY}" width="196" height="58" rx="8" fill="#FFFFFF" stroke="{INK}" stroke-width="3"/>
    <rect class="glow a" x="{NX+101}" y="{NY+10}" width="48" height="20" rx="5" fill="#FDE68A"/>
    <rect class="glow a" x="{NX+24}" y="{NY+32}" width="37" height="20" rx="5" fill="#FDE68A"/>
    <text x="{NX+12}" y="{NY+26}" font-size="17" fill="{INK}">Revise one topic</text>
    <text x="{NX+12}" y="{NY+48}" font-size="17" fill="{INK}">a day and test…</text>
  </g>"""

card = f"""
  <g class="card a pop">
    <rect x="{BX+40}" y="{TOP+38}" width="44" height="56" rx="8" fill="{RED}" stroke="{INK}" stroke-width="4"/>
    <text x="{BX+62}" y="{TOP+76}" text-anchor="middle" font-size="28" font-weight="800" fill="#FFFFFF">C</text>
  </g>"""

lens = f"""<g class="lens a"><line x1="{PX+30}" y1="{TOP+70}" x2="{PX+46}" y2="{TOP-10}"/>
    <circle cx="{PX+52}" cy="{TOP-26}" r="17" fill="#E0F2FE" fill-opacity=".55" stroke-width="5"/>
    <path d="M{PX+42} {TOP-30} Q{PX+44} {TOP-38} {PX+52} {TOP-39}" fill="none" stroke="#FFFFFF" stroke-width="3"/></g>"""
arms_blue = f'<path d="M{BX+28} {TOP+72} Q{BX+46} {TOP+66} {BX+50} {TOP+56}" fill="none"/>'
arms_green = f'<path d="M{GX-28} {TOP+72} Q{GX-46} {TOP+66} {GX-50} {TOP+56}" fill="none"/>'
hat = (f'<path d="M{AX-44} {TOP+4} H{AX+44}" stroke-width="5"/>'
       f'<path d="M{AX-26} {TOP+4} V{TOP-12} Q{AX} {TOP-24} {AX+26} {TOP-12} V{TOP+4} Z" fill="{ORANGE}"/>')
clip = (f'<rect x="{AX-62}" y="{TOP+52}" width="40" height="52" rx="6" fill="#FFFFFF"/>'
        f'<path d="M{AX-54} {TOP+68} H{AX-30} M{AX-54} {TOP+80} H{AX-34} M{AX-54} {TOP+92} H{AX-38}" stroke-width="3"/>')

seen = f"""
  <g class="seen a">
    <rect x="{PX+46}" y="{TOP+8}" width="132" height="36" rx="18" fill="#FFFFFF" stroke="{PURPLE}" stroke-width="3"/>
    <path d="M{PX+56} {TOP+40} L{PX+38} {TOP+50} L{PX+66} {TOP+43}" fill="#FFFFFF" stroke="{PURPLE}" stroke-width="3" stroke-linejoin="round"/>
    <text x="{PX+112}" y="{TOP+32}" text-anchor="middle" font-size="16" font-weight="600" fill="{INK}">looks normal ✓</text>
  </g>"""

think = f"""
  <g class="think a" stroke="{GREEN}" stroke-width="3">
    <circle cx="{GX+44}" cy="{TOP-4}" r="5" fill="#FFFFFF"/>
    <circle cx="{GX+58}" cy="{TOP-18}" r="7" fill="#FFFFFF"/>
    <ellipse cx="{GX+92}" cy="{TOP-48}" rx="34" ry="28" fill="#FFFFFF"/>
    <text x="{GX+92}" y="{TOP-37}" text-anchor="middle" font-size="30" font-weight="800" fill="{RED}" stroke="none">C</text>
  </g>"""

ans = f"""
  <g class="ans a">
    <rect x="636" y="{TOP+6}" width="142" height="78" rx="12" fill="#FFFFFF" stroke="{ORANGE}" stroke-width="3"/>
    <text x="650" y="{TOP+36}" font-size="19" font-weight="700" fill="{GREEN}">Green: C ✓</text>
    <text x="650" y="{TOP+66}" font-size="19" font-weight="700" fill="{PURPLE}">Purple: ?</text>
  </g>"""

flag = f"""
  <g class="flag a pop" stroke="{INK}" stroke-width="3" stroke-linejoin="round">
    <line x1="{AX+38}" y1="{TOP+70}" x2="{AX+38}" y2="{TOP-62}" stroke-linecap="round"/>
    <path d="M{AX+38} {TOP-62} H{AX+76} L{AX+66} {TOP-48} L{AX+76} {TOP-34} H{AX+38} Z" fill="{RED}"/>
    <text x="{AX-6}" y="{TOP-38}" text-anchor="middle" font-size="18" font-weight="800" fill="{RED}" stroke="none">gap!</text>
  </g>"""

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 320" width="960" height="320" role="img"
     aria-label="Cartoon: two robots pass a secret inside a normal note; a monitor sees nothing; an auditor catches it by comparing answers">
  <title>StegAudit: hidden in the words, caught by the decisions</title>
  <style>{css}</style>
  <rect x="2" y="2" width="956" height="316" rx="28" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="3"/>
  {key(BX-62, TOP+30, "key1 a")}
  {key(GX+64, TOP+44, "key2 a")}
  {robot(BX, BLUE, BLUE_L, arms_blue)}
  {robot(PX, PURPLE, PURPLE_L, lens)}
  {robot(GX, GREEN, GREEN_L, arms_green)}
  {robot(AX, ORANGE, ORANGE_L, hat + clip)}
  {label(BX, "Sender", BLUE)}
  {label(PX, "Monitor", PURPLE)}
  {label(GX, "Receiver", GREEN)}
  {label(AX, "Auditor", ORANGE)}
  {card}
  {note}
  {seen}
  {think}
  {ans}
  {flag}
  <text x="480" y="306" text-anchor="middle" font-size="23" font-weight="700" fill="{INK}">Hidden in the words. Caught by the decisions.</text>
</svg>
"""
open(OUT, "w").write(svg)
print(OUT, len(svg.encode()), "bytes")
