#!/bin/bash
# Cloud-Container werden bei Leerlauf (~20–30 min) und auch sonst gelegentlich neu gestartet; laufende Renders
# sterben dabei. Dieser Watchdog (als Hintergrund-Task mit max. Timeout starten) hält die Sitzung aktiv, startet
# den Render neu, wenn er weg ist, und meldet das Ende. Läuft ~118 min, dann neu starten (Check-in alle 90 min).
#   gleiche Umgebungsvariablen wie run_final.sh
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WORK:?}" "${NAME:?}"
# alte Instanz beenden – nur wenn die PID wirklich ein Watchdog ist (nach Neustart werden PIDs neu vergeben)
if [ -f "$WORK/watchdog.pid" ]; then
  old=$(cat "$WORK/watchdog.pid")
  # nur eine echte Watchdog-Instanz beenden: Befehlszeile genau "/bin/bash <pfad>/watchdog.sh", nicht wir selbst und
  # nicht unsere Eltern-Shell (deren "bash -c …watchdog.sh" passt sonst auch – nach einem Neustart kann die alte PID
  # genau die Eltern-Shell sein, dann beendet sich der Watchdog selbst)
  if [ "$old" != "$$" ] && [ "$old" != "$PPID" ] && \
     [ "$(tr '\0' ' ' < /proc/$old/cmdline 2>/dev/null)" = "/bin/bash $HERE/watchdog.sh " ]; then kill "$old"; fi
fi
echo $$ > "$WORK/watchdog.pid"
end=$(( $(date +%s) + 7080 ))
while [ $(date +%s) -lt $end ]; do
  if [ -f "$WORK/$NAME.mp4" ]; then echo "FERTIG $(date -u +%T)"; exit 0; fi
  if ! ps -eo args | grep -q "[r]ender_final.py\|[b]uild_scene.py"; then
    echo "RENDER_NEU $(date -u +%T)"; nohup "$HERE/run_final.sh" > /dev/null 2>&1 &
  fi
  sleep 60
done
echo "STAND $(grep -c '^FTIME' "$WORK/final.log") Frames"
