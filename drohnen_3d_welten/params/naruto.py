POST = dict(
    mist_depth=6000, haze_dist=1600, haze_start=20, haze_color=(0.62, 0.70, 0.82), haze_sky=True,
    look="AgX - Medium High Contrast", sat=1.12, ev=0.15,
    ae_ref=-2.3, ae_strength=0.55,                    # Auto-Belichtung (Video und Standbilder gleich)
    bloom=0.22, bloom_thr=1.6, fog_glow=0.6, bloom_clamp=4.0,   # Glare „Fog Glow“, Quelle begrenzt
    dispersion=0.006,                                 # laterale Dispersion 0,6 %
    vignette=0.22, grain=0.025,                       # Vignette, Filmkorn ~2 % in den Mitteltönen
    # Impact-Blitze (Zeit, EV, Dispersion, Abklingzeit s): kleine Treffer +0,35…0,6 EV, Klimax +1,8 EV
    # (1–2 fast weiße Frames), Dispersion springt kurz mit
    pulses=[(1.15, 0.4, 0.008, 0.06), (4.1, 0.35, 0.006, 0.06), (5.65, 0.35, 0.006, 0.06), (6.85, 0.4, 0.008, 0.06),
            (12.6, 0.5, 0.01, 0.07), (13.15, 0.6, 0.012, 0.08), (15.0, 1.8, 0.02, 0.07)],
    bitrate="18M",                                    # H.264 High, Zwei-Pass 18 Mbit/s (max. 20)
    still_full=True,
)
