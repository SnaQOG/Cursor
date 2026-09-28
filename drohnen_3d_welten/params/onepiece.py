POST = dict(
    mist_depth=6000, haze_dist=1500, haze_start=0, haze_color=(0.50, 0.62, 0.80), haze_sky=True,
    look="AgX - Medium High Contrast", sat=1.15, ev=0.25,
    ae_ref=-2.1, ae_strength=0.55,                    # Auto-Belichtung (Video und Standbilder gleich)
    bloom=0.22, bloom_thr=1.6, fog_glow=0.6, bloom_clamp=4.0,   # Glare „Fog Glow“, Quelle begrenzt
    dispersion=0.006,                                 # laterale Dispersion 0,6 %
    vignette=0.22, grain=0.025,                       # Vignette, Filmkorn ~2 % in den Mitteltönen
    pulses=[(11.25, 0.35, 0.012, 0.18)],              # Impact beim BEAT_DROP: +0,35 EV, Dispersion +1,2 %
    bitrate="18M",                                    # H.264 High, Zwei-Pass 18 Mbit/s (max. 20)
    still_full=True,
)
