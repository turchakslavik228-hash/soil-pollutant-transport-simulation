import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def generate_mesh_rect(Lx, Ly, nx, ny):
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)

    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([x[i], y[j]])
    nodes = np.array(nodes, dtype=float)

    def node_id(i, j):
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_id(i, j)
            n2 = node_id(i + 1, j)
            n3 = node_id(i, j + 1)
            n4 = node_id(i + 1, j + 1)
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def triangle_area(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def shape_gradients(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(A2) < 1e-14:
        raise ValueError("Вироджений трикутник: площа майже нульова.")

    grads = np.array([
        [y2 - y3, x3 - x2],
        [y3 - y1, x1 - x3],
        [y1 - y2, x2 - x1]
    ], dtype=float) / A2
    return grads


def local_matrices(coords, D, vx, vy, sigma, f_source):
    area = triangle_area(coords)
    grads = shape_gradients(coords)
    v = np.array([vx, vy], dtype=float)

    # Дифузійна матриця
    K_diff = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            K_diff[i, j] = D * area * np.dot(grads[i], grads[j])

    # Адвективна матриця
    K_adv = np.zeros((3, 3), dtype=float)
    for j in range(3):
        adv_term = np.dot(v, grads[j])
        for i in range(3):
            K_adv[i, j] = (area / 3.0) * adv_term

    # Реакційна матриця
    M_react = (sigma * area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=float)

    # МАТРИЦЯ МАС
    M_mass = (area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=float)

    F_loc = (f_source * area / 3.0) * np.ones(3, dtype=float)

    K_loc = K_diff + K_adv + M_react
    return K_loc, M_mass, F_loc


def add_neumann_edge(F, node_a, node_b, q_value, nodes):
    xa, ya = nodes[node_a]
    xb, yb = nodes[node_b]
    edge_len = np.sqrt((xb - xa)**2 + (yb - ya)**2)
    Fe = (q_value * edge_len / 2.0) * np.array([1.0, 1.0])
    F[node_a] += Fe[0]
    F[node_b] += Fe[1]


class FEMSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("МСЕ: Динаміка забруднення шаруватого ґрунту")
        self.root.geometry("1300x900")

        self.cb = None
        self.u_history = []

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.root, padding="10", width=350)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.setup_controls()
        self.setup_plot_area()
        self.run_simulation()

    def setup_controls(self):
        ttk.Label(self.control_frame, text="Параметри задачі",
                  font=("Arial", 14, "bold")).pack(pady=(0, 5))

        params_frame = ttk.Frame(self.control_frame)
        params_frame.pack(fill=tk.X, expand=True)
        params_frame.columnconfigure(1, weight=1)

        self.entries = {}

        param_groups = [
            ("Геометрія та Час", {
                "Lx": ("Довжина області (Lx)", 10.0),
                "Ly": ("Висота області (Ly)", 6.0),
                "nx": ("Поділи по осі X (nx)", 30),
                "ny": ("Поділи по осі Y (ny)", 18),
                "Tmax": ("Час моделювання (Tmax)", 5.0),
                "dt": ("Крок за часом (dt)", 0.2),
            }),
            ("Верхній шар ґрунту (y > Ly/2)", {
                "D1": ("Коеф. дифузії (D1)", 1.0),
                "vx1": ("Швидкість по X (vx1)", 1.0),
                "vy1": ("Швидкість по Y (vy1)", -1.5),
            }),
            ("Нижній шар ґрунту (y ≤ Ly/2)", {
                "D2": ("Коеф. дифузії (D2)", 0.1),
                "vx2": ("Швидкість по X (vx2)", 0.2),
                "vy2": ("Швидкість по Y (vy2)", -0.2),
            }),
            ("Фізика та Граничні умови", {
                "sigma": ("Поглинання (σ)", 0.0),
                "f_source": ("Внутрішнє джерело (f)", 0.0),
                "C_top": ("Конц. зверху (C_top)", 1.0),
                "C_bottom": ("Конц. знизу (C_bot)", 0.0),
                "q_left": ("Потік зліва (q_L)", 0.0),
                "q_right": ("Потік справа (q_R)", 0.0)
            })
        ]

        row_idx = 0
        for group_name, group_params in param_groups:
            lbl_header = ttk.Label(params_frame, text=group_name, font=(
                "Arial", 10, "bold"), foreground="#2C3E50")
            lbl_header.grid(row=row_idx, column=0, columnspan=2,
                            sticky="w", pady=(12, 4))
            row_idx += 1

            for key, (label_text, default_val) in group_params.items():
                lbl = ttk.Label(params_frame, text=label_text, anchor="e")
                lbl.grid(row=row_idx, column=0,
                         sticky="e", padx=(0, 10), pady=2)

                entry = ttk.Entry(params_frame, justify="right", width=10)
                entry.insert(0, str(default_val))
                entry.grid(row=row_idx, column=1, sticky="ew", pady=2)
                self.entries[key] = entry
                row_idx += 1

        lbl_source = ttk.Label(
            params_frame, text="Розподіл джерела зверху", anchor="e")
        lbl_source.grid(row=row_idx, column=0, sticky="e",
                        padx=(0, 10), pady=(10, 2))

        self.source_type_var = tk.StringVar(value="Локальне (x<=2)")
        self.source_combo = ttk.Combobox(
            params_frame, textvariable=self.source_type_var, state="readonly", width=12)
        self.source_combo['values'] = ("Рівномірне", "Локальне (x<=2)")
        self.source_combo.grid(row=row_idx, column=1,
                               sticky="ew", pady=(10, 2))

        self.calc_btn = ttk.Button(
            self.control_frame, text="Розрахувати динаміку", command=self.run_simulation)
        self.calc_btn.pack(pady=15, fill=tk.X)

        self.time_label_var = tk.StringVar(value="Час: 0.0 с (Крок 0)")
        ttk.Label(self.control_frame, textvariable=self.time_label_var, font=(
            "Arial", 11, "bold"), foreground="darkred").pack(pady=(5, 0))

        self.time_slider = ttk.Scale(
            self.control_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.time_slider.pack(fill=tk.X, pady=5)
        self.time_slider.state(["disabled"])

        self.status_var = tk.StringVar(value="Готово до розрахунку")
        ttk.Label(self.control_frame, textvariable=self.status_var,
                  foreground="blue", justify="center").pack(pady=5)

    def setup_plot_area(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_inputs(self):
        try:
            inputs = {k: float(v.get()) if k not in ["nx", "ny"] else int(
                v.get()) for k, v in self.entries.items()}
            inputs["source_type"] = self.source_type_var.get()
            if inputs["dt"] <= 0:
                raise ValueError("Крок часу має бути більше 0")
            return inputs
        except ValueError as e:
            messagebox.showerror(
                "Помилка вводу", "Перевірте дані! nx, ny - цілі. dt > 0.")
            return None

    def on_slider_change(self, val):
        if not self.u_history:
            return
        step = int(float(val))
        self.draw_plot(step)

    def run_simulation(self):
        p = self.get_inputs()
        if not p:
            return

        self.status_var.set("Проводиться розрахунок часу...")
        self.root.update()

        try:
            nodes, elements = generate_mesh_rect(
                p["Lx"], p["Ly"], p["nx"], p["ny"])
            n_nodes = len(nodes)
            n_elem = len(elements)

            # Глобальні матриці
            K_global = lil_matrix((n_nodes, n_nodes), dtype=float)
            M_global = lil_matrix((n_nodes, n_nodes), dtype=float)
            F_global = np.zeros(n_nodes, dtype=float)

            # Асемблювання
            for elem in elements:
                coords = nodes[elem]
                yc = np.mean(coords[:, 1])

                if yc > p["Ly"] / 2.0:
                    D, vx, vy = p["D1"], p["vx1"], p["vy1"]
                else:
                    D, vx, vy = p["D2"], p["vx2"], p["vy2"]

                K_loc, M_loc, F_loc = local_matrices(
                    coords, D, vx, vy, p["sigma"], p["f_source"])

                for i_local, i_global in enumerate(elem):
                    F_global[i_global] += F_loc[i_local]
                    for j_local, j_global in enumerate(elem):
                        K_global[i_global, j_global] += K_loc[i_local, j_local]
                        M_global[i_global, j_global] += M_loc[i_local, j_local]

            # Граничні умови Неймана
            tol = 1e-12
            left_nodes = [i for i, (x, y) in enumerate(
                nodes) if abs(x - 0.0) < tol]
            left_nodes = sorted(left_nodes, key=lambda idx: nodes[idx][1])
            for k in range(len(left_nodes) - 1):
                add_neumann_edge(
                    F_global, left_nodes[k], left_nodes[k + 1], p["q_left"], nodes)

            right_nodes = [i for i, (x, y) in enumerate(
                nodes) if abs(x - p["Lx"]) < tol]
            right_nodes = sorted(right_nodes, key=lambda idx: nodes[idx][1])
            for k in range(len(right_nodes) - 1):
                add_neumann_edge(
                    F_global, right_nodes[k], right_nodes[k + 1], p["q_right"], nodes)

            n_steps = int(p["Tmax"] / p["dt"])
            if n_steps < 1:
                n_steps = 1

            u = np.zeros(n_nodes, dtype=float)
            self.u_history = [u.copy()]  # Крок 0

            # Система (M/dt + K) * U_new = (M/dt)*U_old + F
            A_sys = (M_global / p["dt"]) + K_global
            A_sys = A_sys.tocsr()
            M_global = M_global.tocsr()

            # Умови Діріхле
            dirichlet_nodes = {}
            for i, (x, y) in enumerate(nodes):
                if abs(y - 0.0) < tol:
                    dirichlet_nodes[i] = p["C_bottom"]
                elif abs(y - p["Ly"]) < tol:
                    if p["source_type"] == "Локальне (x<=2)":
                        dirichlet_nodes[i] = p["C_top"] if x <= 2.0 else 0.0
                    else:
                        dirichlet_nodes[i] = p["C_top"]

            all_nodes = np.arange(n_nodes)
            fixed = np.array(sorted(dirichlet_nodes.keys()), dtype=int)
            free = np.array(
                [i for i in all_nodes if i not in dirichlet_nodes], dtype=int)

            for step in range(1, n_steps + 1):
                rhs = (M_global @ u) / p["dt"] + F_global

                u_new = np.zeros(n_nodes, dtype=float)
                for idx, val in dirichlet_nodes.items():
                    u_new[idx] = val

                rhs_mod = rhs.copy()
                if len(fixed) > 0:
                    rhs_mod[free] -= A_sys[free][:, fixed] @ u_new[fixed]

                if len(free) > 0:
                    A_ff = A_sys[free][:, free]
                    u_new[free] = spsolve(A_ff, rhs_mod[free])

                u = u_new.copy()
                self.u_history.append(u.copy())

            self.vmin = min([np.min(arr) for arr in self.u_history])
            self.vmax = max([np.max(arr) for arr in self.u_history])
            if self.vmax - self.vmin < 1e-6:
                self.vmax = self.vmin + 1.0

            self.nodes = nodes
            self.elements = elements
            self.current_p = p

            self.time_slider.configure(to=n_steps)
            self.time_slider.state(["!disabled"])
            self.time_slider.set(n_steps)

            self.status_var.set(
                f"Успіх!\nВузлів: {n_nodes}\nТрикутників: {n_elem}\nКроків часу: {n_steps}")

        except Exception as e:
            messagebox.showerror("Помилка розрахунку",
                                 f"Сталася помилка:\n{str(e)}")
            self.status_var.set("Помилка розрахунку")

    def draw_plot(self, step):
        u = self.u_history[step]
        time_t = step * self.current_p["dt"]
        self.time_label_var.set(
            f"Час: {time_t:.1f} с (Крок {step}/{len(self.u_history)-1})")

        self.fig.clf()
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        triang = mtri.Triangulation(
            self.nodes[:, 0], self.nodes[:, 1], self.elements)

        # Графік 1: Сітка та шари
        self.ax1.triplot(triang, 'k-', linewidth=0.3)
        self.ax1.axhline(self.current_p["Ly"] / 2.0, color='red',
                         linestyle='--', linewidth=2, label='Межа шарів ґрунту')
        self.ax1.set_title("Геометрія та розшарування")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")
        self.ax1.legend(loc="upper right")
        self.ax1.axis("equal")

        # Графік 2: Контур
        contour = self.ax2.tricontourf(
            triang, u, levels=25, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.ax2.triplot(triang, color='white', linewidth=0.1, alpha=0.2)
        self.ax2.set_title(f"Розподіл забруднення (t = {time_t:.1f} с)")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("y")
        self.ax2.axis("equal")

        self.cb = self.fig.colorbar(
            contour, ax=self.ax2, label="Концентрація c(x, y)")
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FEMSimulationApp(root)
    root.mainloop()
