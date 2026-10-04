"""Demonstration of the Weibull module on two datasets."""

from reliability_toolkit.plotting.weibull_plot import analyse

# ----------------------------------------------------------------------
# 7.  DEMONSTRATION
# ----------------------------------------------------------------------
if __name__ == '__main__':

    # Capstone data: chilled-water pump drive-end bearings, complete data
    bearing = [3500, 4800, 5900, 6600, 7400, 8100, 9000, 10200, 11800, 14000]

    res = analyse(bearing,
                  title="Chilled-Water Pump Drive-End Bearing - Weibull Analysis",
                  eval_times=[4000, 8760, 12000],
                  savepath="weibull_bearing.png",
                  show=True)

    # Censored example: 7 failures, 5 suspensions
    cam_f = [1100, 1850, 2400, 2950, 3600, 4300, 5100]
    cam_s = [2200, 3000, 4000, 5500, 5500]
    analyse(cam_f, cam_s,
            title="Filling-Machine Cam Follower - Censored Weibull Analysis",
            savepath="weibull_cam.png",
            show=True)