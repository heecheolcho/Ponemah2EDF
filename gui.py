"""
gui.py  --  Ponemah -> EDF Converter (graphical)
================================================
A simple double-click GUI. No DSI software, no .NET, no coding required.
Pick a Ponemah experiment folder, choose channels, click Convert.

Uses only the Python standard library (tkinter) plus numpy + pyedflib.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from convert import convert_experiment
from ponemah_reader import PurePythonPonemahReader, MockPonemahReader, find_experiment_files


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ponemah → EDF Converter")
        self.geometry("720x600")
        self.minsize(660, 540)
        self._log_q: queue.Queue = queue.Queue()
        self.reader = None
        self.subjects = []
        self._build()
        self.after(100, self._drain_log)

    def _build(self):
        pad = {"padx": 8, "pady": 5}

        f1 = ttk.LabelFrame(self, text="1.  Ponemah experiment folder "
                                       "(contains .PnmWav, .PnmExp, .PnmMarks)")
        f1.pack(fill="x", **pad)
        self.exp_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self.exp_var).pack(side="left", fill="x",
                                                      expand=True, padx=6, pady=6)
        ttk.Button(f1, text="Browse…", command=self._pick_exp).pack(side="left", padx=6)

        bar = ttk.Frame(self); bar.pack(fill="x", **pad)
        ttk.Button(bar, text="Load channels", command=self._load).pack(side="left")
        self.demo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Demo (synthetic, no data needed)",
                        variable=self.demo_var).pack(side="left", padx=10)

        f3 = ttk.LabelFrame(self, text="2.  Channels to export (none selected = all)")
        f3.pack(fill="both", expand=True, **pad)
        self.chan_list = tk.Listbox(f3, selectmode="multiple", exportselection=False)
        self.chan_list.pack(fill="both", expand=True, padx=6, pady=6)

        f4 = ttk.LabelFrame(self, text="3.  Options")
        f4.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value="per_segment")
        ttk.Radiobutton(f4, text="One EDF per acquisition segment (recommended)",
                        variable=self.mode_var, value="per_segment").pack(anchor="w", padx=6)
        ttk.Radiobutton(f4, text="Single concatenated EDF (gaps → annotations)",
                        variable=self.mode_var, value="single").pack(anchor="w", padx=6)
        row = ttk.Frame(f4); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Split into EDFs of N minutes (0 = one file):").pack(side="left", padx=6)
        self.chunk_var = tk.StringVar(value="60")
        ttk.Entry(row, textvariable=self.chunk_var, width=8).pack(side="left")
        ttk.Label(row, text="   Limit to first N seconds (blank = all):").pack(side="left", padx=6)
        self.max_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.max_var, width=10).pack(side="left")

        of = ttk.Frame(f4); of.pack(fill="x", pady=3)
        ttk.Label(of, text="Output folder:").pack(side="left", padx=6)
        self.out_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "EDF_output"))
        ttk.Entry(of, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(of, text="Browse…", command=self._pick_out).pack(side="left", padx=6)

        f5 = ttk.Frame(self); f5.pack(fill="x", **pad)
        self.convert_btn = ttk.Button(f5, text="Convert to EDF", command=self._convert)
        self.convert_btn.pack(side="left", padx=6)
        self.prog = ttk.Progressbar(f5, mode="indeterminate")
        self.prog.pack(side="left", fill="x", expand=True, padx=6)

        self.log = tk.Text(self, height=9, wrap="word")
        self.log.pack(fill="both", expand=False, padx=8, pady=6)
        self._logmsg("Ready. Pick an experiment folder and click 'Load channels'.")

    def _pick_exp(self):
        d = filedialog.askdirectory(title="Select Ponemah experiment folder")
        if d:
            self.exp_var.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)

    def _logmsg(self, m):
        self._log_q.put(str(m))

    def _drain_log(self):
        while not self._log_q.empty():
            self.log.insert("end", self._log_q.get() + "\n")
            self.log.see("end")
        self.after(100, self._drain_log)

    def _max_seconds(self):
        t = self.max_var.get().strip()
        try:
            return float(t) if t else None
        except ValueError:
            return None

    def _chunk_seconds(self):
        t = self.chunk_var.get().strip()
        try:
            v = float(t) if t else 0
        except ValueError:
            v = 0
        return v * 60 if v > 0 else None

    def _make_reader(self):
        if self.demo_var.get():
            return MockPonemahReader()
        exp = self.exp_var.get().strip()
        if not exp or not find_experiment_files(exp)[0]:
            raise ValueError("Pick a valid experiment folder (must contain a .PnmWav file).")
        return PurePythonPonemahReader(exp, max_seconds=self._max_seconds())

    def _load(self):
        try:
            self.reader = self._make_reader()
            self.subjects = self.reader.list_subjects()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load failed", str(e)); return
        self.chan_list.delete(0, "end")
        seen = []
        for s in self.subjects:
            for c in s.channels:
                tag = f"{c.label} ({c.sample_rate:g} Hz, {c.unit})"
                if tag not in seen:
                    seen.append(tag); self.chan_list.insert("end", tag)
        names = ", ".join(s.name for s in self.subjects)
        self._logmsg(f"Loaded subject(s): {names}. {len(seen)} channel(s). "
                     f"Select channels (or none for all) and Convert.")

    def _convert(self):
        if not self.reader:
            messagebox.showwarning("No data", "Click 'Load channels' first."); return
        # rebuild reader to pick up any change to the max-seconds box
        try:
            self.reader = self._make_reader()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e)); return
        chan_sel = [self.chan_list.get(i).split(" (")[0]
                    for i in self.chan_list.curselection()]
        out_dir = self.out_var.get().strip() or os.path.join(os.path.expanduser("~"), "EDF_output")
        mode = self.mode_var.get()
        self.convert_btn.config(state="disabled"); self.prog.start(12)

        def worker():
            try:
                paths = convert_experiment(self.exp_var.get() or "(demo)", out_dir,
                                           self.reader, mode=mode,
                                           channel_labels=chan_sel or None,
                                           progress=self._logmsg,
                                           chunk_seconds=self._chunk_seconds())
                self._logmsg(f"DONE. {len(paths)} file(s) saved to: {out_dir}")
                messagebox.showinfo("Conversion complete",
                                    f"{len(paths)} EDF file(s) saved to:\n{out_dir}")
            except Exception as e:  # noqa: BLE001
                self._logmsg(f"ERROR: {e}")
                messagebox.showerror("Conversion failed", str(e))
            finally:
                self.prog.stop(); self.convert_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
