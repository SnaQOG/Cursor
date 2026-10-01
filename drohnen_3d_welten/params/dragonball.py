POST = dict(mist_depth=6000, haze_dist=1500, haze_start=30, haze_color=(0.66, 0.78, 0.62), haze_sky=True,
            look="AgX - Medium High Contrast", sat=1.0, ev=0.0, ae_strength=0.55,
            bloom=0.25, bloom_thr=1.8, fog_glow=0.6, bloom_clamp=4.0, dispersion=0.006, vignette=0.22, grain=0.025,
            pulses=[(10.0, 0.4, 0.008, 0.06), (11.11, 0.2, 0.004, 0.05), (11.47, 0.3, 0.006, 0.06), (11.95, 0.35, 0.006, 0.06),
                    (12.15, 0.7, 0.012, 0.09), (13.25, 0.4, 0.008, 0.07), (13.5, 0.4, 0.008, 0.07),
                    (15.05, 0.3, 0.006, 0.1), (15.8, 1.8, 0.02, 0.08)],
            flare=dict(t0=15.75, t1=17.2, thr=6.0, strength=0.6, streak=0.5, fade_out=0.8),
            haze_fade=[(15.6, 1.0), (16.0, 0.0), (21.0, 0.0)],
            bitrate="18M", still_full=True)
