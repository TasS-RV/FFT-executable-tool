import sys, time, os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from filter_data import *

# Make Tk report exceptions to console (more readable)
def _tk_exception_handler(self, exc, val, tb):
    import traceback
    print("\n--- Tkinter exception ---")
    traceback.print_exception(exc, val, tb)
tk.Tk.report_callback_exception = _tk_exception_handler


class FFTToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Analysis Tool")
        self.root.geometry("1400x820")

        # Data & state
        self.full_df = None
        self.numeric_cols = []
        self.original_dx = None

        # Crosshair / annotation handles
        self.vline_raw = None
        self.hline_raw = None
        self.vline_fft = None
        self.hline_fft = None
        self.fft_text = None

        # --- Left control frame ---
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side="left", fill="y")

        ttk.Label(ctrl, text="File & Columns", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(ctrl, text="Load CSV / Excel", command=self.load_file).pack(fill="x", pady=(6, 8))
        self.file_label = ttk.Label(ctrl, text="No file loaded", wraplength=260)
        self.file_label.pack(anchor="w", pady=(0, 6))

        ttk.Label(ctrl, text="X column (time/pos):").pack(anchor="w")
        self.x_cb = ttk.Combobox(ctrl, state="readonly", width=30)
        self.x_cb.pack(anchor="w", pady=2)
        self.x_cb.bind("<<ComboboxSelected>>", self.on_col_change)

        ttk.Label(ctrl, text="Y column (signal):").pack(anchor="w", pady=(6, 0))
        self.y_cb = ttk.Combobox(ctrl, state="readonly", width=30)
        self.y_cb.pack(anchor="w", pady=2)
        self.y_cb.bind("<<ComboboxSelected>>", self.on_col_change)

        ttk.Separator(ctrl).pack(fill="x", pady=8)

        # step multiplier - integer Spinbox (explicit integers)
        ttk.Label(ctrl, text="Step multiplier (subsample every Nth point):").pack(anchor="w")
        self.step_var = tk.IntVar(value=1)
        self.step_spin = tk.Spinbox(ctrl, from_=1, to=50, textvariable=self.step_var, width=8,
                                    command=self.on_step_change)
        self.step_spin.pack(anchor="w", pady=(2, 6))
        ttk.Label(ctrl, text="(Integer; 1 = use all samples)").pack(anchor="w", pady=(0,6))

        # FFT range controls
        ttk.Label(ctrl, text="FFT frequency limits (optional)").pack(anchor="w", pady=(6, 0))
        fr_frame = ttk.Frame(ctrl)
        fr_frame.pack(anchor="w", pady=2)
        ttk.Label(fr_frame, text="Fmin").grid(row=0, column=0)
        self.fmin_entry = ttk.Entry(fr_frame, width=10); self.fmin_entry.grid(row=0, column=1, padx=(4,10))
        ttk.Label(fr_frame, text="Fmax").grid(row=0, column=2)
        self.fmax_entry = ttk.Entry(fr_frame, width=10); self.fmax_entry.grid(row=0, column=3, padx=(4,0))

        # Y limits for raw and FFT
        ttk.Label(ctrl, text="Raw Y-limits (optional)").pack(anchor="w", pady=(8,0))
        raw_frame = ttk.Frame(ctrl)
        raw_frame.pack(anchor="w", pady=2)
        ttk.Label(raw_frame, text="Ymin").grid(row=0, column=0); self.raw_ymin = ttk.Entry(raw_frame, width=8); self.raw_ymin.grid(row=0,column=1,padx=4)
        ttk.Label(raw_frame, text="Ymax").grid(row=0, column=2); self.raw_ymax = ttk.Entry(raw_frame, width=8); self.raw_ymax.grid(row=0,column=3,padx=4)

        ttk.Label(ctrl, text="FFT Y-limits (optional)").pack(anchor="w", pady=(6,0))
        fftyl = ttk.Frame(ctrl)
        fftyl.pack(anchor="w", pady=2)
        ttk.Label(fftyl, text="Ymin").grid(row=0, column=0); self.fft_ymin = ttk.Entry(fftyl, width=8); self.fft_ymin.grid(row=0,column=1,padx=4)
        ttk.Label(fftyl, text="Ymax").grid(row=0, column=2); self.fft_ymax = ttk.Entry(fftyl, width=8); self.fft_ymax.grid(row=0,column=3,padx=4)

        # Run & Save controls
        ttk.Button(ctrl, text="Run (recompute)", command=self.run_plot).pack(fill="x", pady=(12,6))
        ttk.Button(ctrl, text="Save Snapshot", command=self.save_snapshot).pack(fill="x")

        ttk.Separator(ctrl).pack(fill="x", pady=10)
        self.cursor_label = ttk.Label(ctrl, text="Cursor: (none)", wraplength=260)
        self.cursor_label.pack(anchor="w", pady=(6,0))

        # --- Right plot area ---
        self.fig, (self.ax_raw, self.ax_fft) = plt.subplots(2, 1, figsize=(9, 7), dpi=100)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="right", fill="both", expand=True)

        # Connect mouse motion for crosshair
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Defensive: prevent on_mouse_move from running before plots exist
        self.plotted_once = False

    # -------------------------
    # File loading / detection
    # -------------------------
    def load_file(self):
        path = filedialog.askopenfilename(title="Select CSV or Excel file",
                                          filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx;*.xls")])
        if not path:
            return
        self.file_label.config(text=os.path.basename(path))
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path, header=0, low_memory=False)
            else:
                df = pd.read_excel(path, header=0)
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to read file:\n{e}")
            return

        # detect numeric-like columns (keep any column that has at least one numeric cell)
        numeric_like = []
        for c in df.columns:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                numeric_like.append(c)

        if not numeric_like:
            messagebox.showerror("No numeric columns", "No columns with numeric content detected.")
            return

        self.full_df = df
        self.numeric_cols = numeric_like
        self.x_cb['values'] = numeric_like
        self.y_cb['values'] = numeric_like
        # sensible defaults
        self.x_cb.set(numeric_like[0])
        self.y_cb.set(numeric_like[1] if len(numeric_like) > 1 else numeric_like[0])

        print(f"Loaded file: {os.path.basename(path)}")
        print("Numeric-like columns:", numeric_like)

    # -------------------------
    # Column change
    # -------------------------
    def on_col_change(self, event=None):
        if self.full_df is None:
            return
        col = event.widget.get()
        print(f"Selected column: {col}")
        if col in self.full_df.columns:
            print(self.full_df[col].head(10))

    def on_step_change(self, *args):
        """
        This function is used to change the step_size multipler - note that for the sine and cosine data, the between points the 'angle' step size
        is not constant (as mechanical speed is not strictly constant).

        - Step size increments are forced to integer multiplers of the base step size (dx_base).
        - If the user manually types into the Spinbox, the integer constraint is enforced.
        """
        try:
            v = int(self.step_var.get())
            if v < 1:
                self.step_var.set(1)
        except Exception:
            self.step_var.set(1)
        # do not auto-run here to avoid unintended heavy recompute

    # -------------------------
    # Run plotting + FFT
    # -------------------------
    def run_plot(self):
        if self.full_df is None:
            messagebox.showwarning("No data", "Please load a CSV/Excel file first.")
            return

        x_col = self.x_cb.get()
        y_col = self.y_cb.get()
        if not x_col or not y_col:
            messagebox.showwarning("Select columns", "Select both X and Y columns.")
            return

        # coerce both columns to numeric and drop rows where either is NaN
        xs = pd.to_numeric(self.full_df[x_col], errors='coerce')
        ys = pd.to_numeric(self.full_df[y_col], errors='coerce')
        valid = xs.notna() & ys.notna()
        if valid.sum() < 4:
            messagebox.showerror("Not enough numeric rows", "Need at least 4 rows with numeric X and Y.")
            return

        x_all = xs[valid].to_numpy()
        y_all = ys[valid].to_numpy()

        # Sorting x-data: if input data somehow has non chronological time, or picks up 'reverse' rotations  is an error!).
        if not np.all(np.diff(x_all) >= 0):
            order = np.argsort(x_all)
            x_all = x_all[order]
            y_all = y_all[order]

        """
        ANCHOR| Used for interpolating data to uniform grid and sub-sampling.

        1. dx_base = float(np.mean(dxs)) --> This is the mean step size, used to create a uniform x-grid for FFT.
           FFT requires uniformly spaced data, so we interpolate y-values onto this uniform grid.

        2. x_uniform, y_interp --> Create uniform x-grid and interpolate y-values onto it using np.interp().
           This ensures the FFT receives properly spaced data matching the frequency axis.

        3. mult = int(self.step_var.get()) --> This is the step size multiplier, used to sub-sample the interpolated data.
           x_sub = x_uniform[::mult] --> Sub-sampled uniform x data.
           y_sub = y_interp[::mult] --> Sub-sampled interpolated y data.

        4. n = len(y_sub) --> Number of points in the sub-sampled interpolated data.
           The FFT is computed on these interpolated y-values, which are now properly spaced.
        
        """
        # compute base sampling interval (original dx) using diffs of full data
        dxs = np.diff(x_all)
        dxs = dxs[np.isfinite(dxs) & (dxs > 0)]  # Remove any infinite or negative values
        if dxs.size == 0: 
            messagebox.showerror("Bad x spacing", "Cannot compute a valid sampling interval from X.")
            return
        dx_base = float(np.mean(dxs))
        self.original_dx = dx_base

        """
        FILTERING STAGE: applied on raw data rather than interpolated data, to avoid losing any information.

        Functions have been imported from filter_data.py to avoid clutter.
        
        Justifications for frequency cut-offs, roll-offs etc... are defined below for each different filtering stage.
        """

        y_clean = preprocess_signal(x_all, y_all,
                                    do_median=True, median_k=5,
                                    do_hp=True, hp_cut=0.1,
                                    do_notch=True, notch_freqs=[50.0], notch_Q=30,
                                    do_lp=True, lp_cut=250.0)
        # Resetting the cleaned y_all data:
        y_all = y_clean

        # Create uniform x-grid based on mean step size for FFT (FFT requires uniform spacing)
        x_min = float(x_all[0])
        x_max = float(x_all[-1])
        x_uniform = np.arange(x_min, x_max + dx_base, dx_base)
        # Ensure we don't exceed the original range
        x_uniform = x_uniform[x_uniform <= x_max]
        
        """
        ANCHOR|Critical step: Interpolate the y-data to a uniform list of x-values - increments are dx_base. This does not affect the multiplier, because it will not affect the step_size multipler,
        given it is index based (rather than being value based, so will land on interpolated y-values anyways).

        Note: this still captures any spiky, horrible noisy datapoints.
        """
        y_interp = np.interp(x_uniform, x_all, y_all)
        
        # integer subsampling multiplier from Spinbox
        try:
            mult = int(self.step_var.get())
            if mult < 1:
                mult = 1
                self.step_var.set(1)
        except Exception:
            mult = 1
            self.step_var.set(1)

        # Subsample the interpolated data by skipping every (mult-1) samples: take every mult'th sample
        x_sub = x_uniform[::mult]
        y_sub = y_interp[::mult]

        if x_sub.size < 4:
            messagebox.showerror("Too few subsampled points",
                                 "Step multiplier too large for this dataset (not enough points).")
            return

        # Compute FFT on subsampled interpolated y_sub with frequency axis based on dx_base
        # The y-values are now properly interpolated onto a uniform grid
        n = len(y_sub)
        y_dm = y_sub - np.mean(y_sub) #--> This is the mean-centered y data. This is done to remove the DC component from the data, which is not relevant to the FFT removing a massive low frequency peak.
        fft_vals = np.fft.fft(y_dm)
        # use dx_base for frequency axis (now correctly matches the uniform spacing)
        freq = np.fft.fftfreq(n, d=dx_base)
        amp = np.abs(fft_vals) / n

        # Keep positive freqs - apply mask to filter out negaitve frequency values
        pos_mask = freq >= 0
        freq_pos = freq[pos_mask]
        amp_pos = amp[pos_mask]

        # Apply optional frequency plotting limits - in order to scale and truncate the graphs to a particular range
        try:
            fmin = float(self.fmin_entry.get()) if self.fmin_entry.get().strip() != "" else None
        except ValueError:
            fmin = None
        try:
            fmax = float(self.fmax_entry.get()) if self.fmax_entry.get().strip() != "" else None
        except ValueError:
            fmax = None

        mask = np.ones_like(freq_pos, dtype=bool)
        if fmin is not None:
            mask &= (freq_pos >= fmin)
        if fmax is not None:
            mask &= (freq_pos <= fmax)
        freq_plot = freq_pos[mask]
        amp_plot = amp_pos[mask]

        # --- Plot raw data (top) ---
        # Show the interpolated uniform data used for FFT
        self.ax_raw.clear()
        if mult > 1:
            # Plot the subsampled interpolated data that was used for FFT
            self.ax_raw.plot(x_sub, y_sub, lw=0.8, color="tab:blue", 
                           label=f"Subsampled interpolated (mult={mult})")
            # Show original non-uniform data in lighter color for reference
            self.ax_raw.plot(x_all, y_all, lw=0.3, color="gray", alpha=0.3, 
                           label="Original (non-uniform)")
            self.ax_raw.legend(fontsize=8)
        else:
            # When mult=1, show interpolated uniform data (what FFT uses)
            self.ax_raw.plot(x_uniform, y_interp, lw=0.8, color="tab:blue", 
                           label="Interpolated (uniform grid)")
            # Show original non-uniform data in lighter color for reference
            self.ax_raw.plot(x_all, y_all, lw=0.3, color="gray", alpha=0.3, 
                           marker='o', markersize=2, label="Original (non-uniform)")
            self.ax_raw.legend(fontsize=8)
        self.ax_raw.set_title(f"Raw data: {y_col} vs {x_col}")
        self.ax_raw.set_xlabel(x_col)
        self.ax_raw.set_ylabel(y_col)
        self.ax_raw.grid(True)

        # apply raw Y limits if given
        try:
            rymin = float(self.raw_ymin.get()) if self.raw_ymin.get().strip() != "" else None
            rymax = float(self.raw_ymax.get()) if self.raw_ymax.get().strip() != "" else None
            if rymin is not None or rymax is not None:
                self.ax_raw.set_ylim(rymin, rymax)
        except ValueError:
            pass

        # --- Plot FFT (bottom) ---
        self.ax_fft.clear()
        self.ax_fft.plot(freq_plot, amp_plot, lw=0.8, color="orange")
        self.ax_fft.set_title("FFT Spectrum")
        self.ax_fft.set_xlabel("Frequency (1 / X unit)")
        self.ax_fft.set_ylabel("Amplitude")
        self.ax_fft.grid(True)

        # apply FFT Y limits if given
        try:
            fymin = float(self.fft_ymin.get()) if self.fft_ymin.get().strip() != "" else None
            fymax = float(self.fft_ymax.get()) if self.fft_ymax.get().strip() != "" else None
            if fymin is not None or fymax is not None:
                self.ax_fft.set_ylim(fymin, fymax)
        except ValueError:
            pass

        # initialize crosshair lines as scalar positions (axvline expects scalar on creation)
        # They will later be updated with set_xdata([x,x]) sequences on mouse move
        if self.vline_raw is None:
            x0 = (self.ax_raw.get_xlim()[0] + self.ax_raw.get_xlim()[1]) / 2.0
            y0 = (self.ax_raw.get_ylim()[0] + self.ax_raw.get_ylim()[1]) / 2.0
            self.vline_raw = self.ax_raw.axvline(x0, color="gray", linestyle="--", lw=0.8)
            self.hline_raw = self.ax_raw.axhline(y0, color="gray", linestyle="--", lw=0.8)
        if self.vline_fft is None:
            xf0 = (self.ax_fft.get_xlim()[0] + self.ax_fft.get_xlim()[1]) / 2.0
            yf0 = (self.ax_fft.get_ylim()[0] + self.ax_fft.get_ylim()[1]) / 2.0
            self.vline_fft = self.ax_fft.axvline(xf0, color="gray", linestyle="--", lw=0.8)
            self.hline_fft = self.ax_fft.axhline(yf0, color="gray", linestyle="--", lw=0.8)

        # small annotation in FFT axes for dynamic text
        if self.fft_text is None:
            self.fft_text = self.ax_fft.text(0.02, 0.95, "", transform=self.ax_fft.transAxes, fontsize=9,
                                             verticalalignment="top")

        # draw
        self.canvas.draw_idle()
        self.plotted_once = True

        # print small summary to terminal
        print(f"Run: dx_base={dx_base:.6g}, multiplier={mult}, subsampled_points={n}")
        print("First positive freq bins (up to 6):")
        for f, a in zip(freq_pos[:6], amp_pos[:6]):
            print(f"  f={f:.6g}, amp={a:.6g}")

    # -------------------------
    # Mouse move crosshair
    # -------------------------
    def on_mouse_move(self, event):
        # don't do anything before first plot
        if not self.plotted_once:
            return
        if event.inaxes is None:
            self.cursor_label.config(text="Cursor: outside axes")
            return

        x = event.xdata
        y = event.ydata
        ax = event.inaxes

        if x is None or y is None:
            return

        # update crosshairs; set_xdata/set_ydata expect sequences
        try:
            if ax == self.ax_raw:
                if self.vline_raw is not None:
                    self.vline_raw.set_xdata([x, x])
                    self.hline_raw.set_ydata([y, y])
                # also move the FFT vertical to the same x (optional)
                if self.vline_fft is not None:
                    self.vline_fft.set_xdata([x, x])
                # update labels
                self.cursor_label.config(text=f"Raw: x={x:.6g}, y={y:.6g}")
                if self.fft_text is not None:
                    self.fft_text.set_text(f"x={x:.6g}")
            elif ax == self.ax_fft:
                if self.vline_fft is not None:
                    self.vline_fft.set_xdata([x, x])
                    self.hline_fft.set_ydata([y, y])
                # reflect same x on raw vertical line
                if self.vline_raw is not None:
                    self.vline_raw.set_xdata([x, x])
                self.cursor_label.config(text=f"FFT: f={x:.6g}, A={y:.6g}")
                if self.fft_text is not None:
                    self.fft_text.set_text(f"f={x:.6g}, A={y:.6g}")
        except Exception as e:
            # defensive: print but don't crash GUI
            print("Cursor update error:", e)

        self.canvas.draw_idle()

    # -------------------------
    # Save snapshot
    # -------------------------
    def save_snapshot(self):
        if not self.plotted_once:
            messagebox.showwarning("Nothing to save", "Plot something first.")
            return
        fn = f"fft_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = filedialog.asksaveasfilename(initialfile=fn, defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Snapshot saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = FFTToolApp(root)
    root.mainloop()
