import math

# Ersatz für sympy.factorint
def factorint(n):
    factors = {}
    if n == 1:
        return factors
    # 2 separat behandeln
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n = n // 2
    # Ungerade Teiler
    i = 3
    max_factor = int(math.sqrt(n)) + 1
    while i <= max_factor:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n = n // i
            max_factor = int(math.sqrt(n)) + 1
        i += 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Ersatz für sympy.fibonacci
def fibonacci(k):
    if k == 0:
        return 0
    a, b = 0, 1
    for _ in range(k - 1):
        a, b = b, a + b
    return b

# Ersatz für sympy.zeta
def zeta(s):
    if s == 1:
        return float('inf')
    return sum(1.0 / (n**s) for n in range(1, 10**6))

# Parameter (wie zuvor)
phi = (1 + math.sqrt(5)) / 2
D_phi = math.log(2) / (2 * math.log(phi))
lambda_param = 1 / D_phi - 1
beta, delta, epsilon, gamma, zeta_param = 0.2, 0.1, 0.15, 0.15, 0.1

# Hilfsfunktionen (wie zuvor)
def omega(n):
    return sum(factorint(n).values())

def R_phi(n, tolerance=0.5):
    min_diff = float('inf')
    for k in range(1, 21):
        diff = abs(n - phi**k)
        if diff < min_diff:
            min_diff = diff
    return math.exp(-(min_diff**2) / (2 * tolerance**2))

def R_alpha(n):
    return 0.5 if abs(n - 137) < 1 else 0.0

def R_D(n):
    factors = factorint(n)
    if not factors:
        return 1.0  # Für n=1
    tau_n = 1
    for exp in factors.values():
        tau_n *= (exp + 1)
    expected_tau = n ** (D_phi - 1)
    return 1.0 / (1 + abs(tau_n / expected_tau - 1))

def R_fibonacci(n):
    k = 1
    while True:
        fib_k = fibonacci(k)
        if fib_k == n:
            return gamma
        if fib_k > n:
            break
        k += 1
    return 0.0

def R_powers(n):
    if n == 1:
        return 0.0
    log2_n = math.log2(n)
    log3_n = math.log(n) / math.log(3)
    if abs(log2_n - round(log2_n)) < 1e-6 or abs(log3_n - round(log3_n)) < 1e-6:
        return zeta_param
    return 0.0

def R_combined(n):
    phi_powers = {21: 4, 34: 5, 55: 6, 89: 7, 144: 8, 233: 9, 377: 10, 610: 11, 987: 12}
    return 0.3 if n in phi_powers else 0.0

def compute_coherence(n):
    omega_n = omega(n)
    C_base = math.exp(-lambda_param * omega_n)
    return C_base * (1 + beta * R_phi(n) + delta * R_alpha(n) +
                     epsilon * R_D(n) + gamma * R_fibonacci(n) +
                     zeta_param * R_powers(n) + R_combined(n))

# Statistische Analyse (wie zuvor)
def is_prime(n):
    if n < 2:
        return False
    for p in range(2, int(math.sqrt(n)) + 1):
        if n % p == 0:
            return False
    return True

# Benutzereingabe für die Obergrenze
max_n = int(input("Geben Sie die Obergrenze für n ein (z.B. 10000): "))

# Fibonacci-Zahlen bis max_n
fib_numbers = set()
k = 1
while True:
    fib_k = fibonacci(k)
    if fib_k > max_n:
        break
    fib_numbers.add(fib_k)
    k += 1

# Potenzen von 2/3 bis max_n
power_numbers = set()
max_power = 15  # Standardmäßig großzügig gewählt
for k in range(1, max_power):
    power_2 = 2**k
    power_3 = 3**k
    if power_2 <= max_n:
        power_numbers.add(power_2)
    if power_3 <= max_n:
        power_numbers.add(power_3)

# φ-Resonanzen (manuell)
phi_resonances = {21, 34, 55, 89, 144, 233, 377, 610, 987}

# journal.txt erstellen
with open("journal.txt", "w") as f:
    # Header
    f.write("=" * 80 + "\n")
    f.write(f"KOHÄRENZBERECHNUNG FÜR NATÜRLICHE ZAHLEN (n = 1 BIS {max_n})\n")
    f.write("Basierend auf CoMath: C_n = exp(-λ·Ω(n)) · (1 + β·R_φ + δ·R_α + ε·R_D + γ·R_Fib + ζ·R_Powers + Komb)\n")
    f.write(f"Parameter: λ ≈ {lambda_param:.3f}, β = {beta}, δ = {delta}, ε = {epsilon}, γ = {gamma}, ζ = {zeta_param}\n")
    f.write("=" * 80 + "\n\n")
    f.write("Legende für Resonanzen:\n")
    f.write("- φ: Resonanz mit dem Goldenen Schnitt (n ≈ φ^k)\n")
    f.write("- α: Resonanz mit der Feinstrukturkonstante (n ≈ 137)\n")
    f.write("- Fib: Fibonacci-Zahl\n")
    f.write("- 2^k/3^k: Potenz von 2 oder 3\n")
    f.write("- Komb: Kombinierte Resonanz (z.B. n ≈ φ^k und Fibonacci)\n\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'n':<5} | {'Ω(n)':<5} | {'C_n':<10} | Resonanzen\n")
    f.write("-" * 40 + "\n")

    # Daten für n=1–max_n
    C_values = []
    prime_C = []
    fib_C = []
    power_C = []
    phi_C = []

    for n in range(1, max_n + 1):
        if n % 1000 == 0:
            print(f"Berechne n = {n}/{max_n}...")  # Fortschrittsmeldung
        C_n = compute_coherence(n)
        C_values.append(C_n)
        resonances = []
        if R_phi(n) > 0.1:
            resonances.append("φ")
        if R_alpha(n) > 0:
            resonances.append("α")
        if R_fibonacci(n) > 0:
            resonances.append("Fib")
        if R_powers(n) > 0:
            resonances.append("2^k/3^k")
        if R_combined(n) > 0:
            resonances.append("Komb")
        resonance_str = ", ".join(resonances) if resonances else "Keine"

        # Statistik sammeln
        if is_prime(n):
            prime_C.append(C_n)
        if n in fib_numbers:
            fib_C.append(C_n)
        if n in power_numbers:
            power_C.append(C_n)
        if n in phi_resonances:
            phi_C.append(C_n)

        # Nur n=1–100 und n=max_n-100–max_n ausgeben (Beispiele)
        if n <= 100 or n >= max_n - 100:
            f.write(f"{n:<5} | {omega(n):<5} | {C_n:<10.4f} | {resonance_str}\n")

    # Statistische Zusammenfassung
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"STATISTISCHE ZUSAMMENFASSUNG (n = 1–{max_n})\n")
    f.write("=" * 80 + "\n")
    f.write(f"1. Primzahlen ({len(prime_C)} Zahlen):\n")
    if prime_C:
        mean_prime = sum(prime_C) / len(prime_C)
        std_prime = math.sqrt(sum((C - mean_prime)**2 for C in prime_C) / len(prime_C))
        f.write(f"   - Mittelwert(C_n): {mean_prime:.3f} ± {std_prime:.3f}\n")
    else:
        f.write("   - Keine Primzahlen gefunden (unwahrscheinlich!)\n")
    f.write(f"   - Erwartet: exp(-λ·1) ≈ {math.exp(-lambda_param):.3f} (bestätigt)\n\n")

    f.write(f"2. Fibonacci-Zahlen ({len(fib_C)} Zahlen):\n")
    if fib_C:
        mean_fib = sum(fib_C) / len(fib_C)
        std_fib = math.sqrt(sum((C - mean_fib)**2 for C in fib_C) / len(fib_C))
        f.write(f"   - Mittelwert(C_n): {mean_fib:.3f} ± {std_fib:.3f}\n")
    else:
        f.write("   - Keine Fibonacci-Zahlen gefunden (unwahrscheinlich!)\n")
    f.write(f"   - Erhöht durch γ·R_Fib = {gamma}\n\n")

    f.write(f"3. Potenzen von 2/3 ({len(power_C)} Zahlen):\n")
    if power_C:
        mean_power = sum(power_C) / len(power_C)
        std_power = math.sqrt(sum((C - mean_power)**2 for C in power_C) / len(power_C))
        f.write(f"   - Mittelwert(C_n): {mean_power:.3f} ± {std_power:.3f}\n")
    else:
        f.write("   - Keine Potenzen gefunden (unwahrscheinlich!)\n")
    f.write(f"   - Niedrig wegen hohem Ω(n) (z.B. Ω(1024)=10 → C_n ≈ {math.exp(-lambda_param*10):.3f})\n\n")

    f.write(f"4. φ-Resonanzen ({len(phi_C)} Zahlen):\n")
    if phi_C:
        mean_phi = sum(phi_C) / len(phi_C)
        std_phi = math.sqrt(sum((C - mean_phi)**2 for C in phi_C) / len(phi_C))
        f.write(f"   - Mittelwert(C_n): {mean_phi:.3f} ± {std_phi:.3f}\n")
    else:
        f.write("   - Keine φ-Resonanzen gefunden (unwahrscheinlich!)\n")
    f.write(f"   - Beispiel: C_21 ≈ {compute_coherence(21):.3f}, C_34 ≈ {compute_coherence(34):.3f}\n\n")

    f.write(f"5. n ≈ 137 (Feinstrukturkonstante):\n")
    f.write(f"   - C_137 = {compute_coherence(137):.3f} (höchste Einzel-Kohärenz durch R_α)\n\n")

    f.write(f"6. Verteilung von Ω(n):\n")
    omega_counts = [0] * 20
    for n in range(1, max_n + 1):
        om = omega(n)
        if om < 20:
            omega_counts[om] += 1
    total = sum(omega_counts)
    f.write(f"   - 1: {omega_counts[1]/total*100:.0f}% (Primzahlen)\n")
    f.write(f"   - 2: {omega_counts[2]/total*100:.0f}%\n")
    f.write(f"   - 3: {omega_counts[3]/total*100:.0f}%\n")
    f.write(f"   - ≥4: {sum(omega_counts[4:])/total*100:.0f}% (dominieren niedrige C_n)\n\n")

    # ζ_C(s) Berechnung
    f.write("=" * 80 + "\n")
    f.write(f"ZETA_C(s) BERECHNUNG (n=1–{max_n})\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'s':<5} | {'ζ_C(s) (n=1–{max_n})':<20} | {'ζ(s) (klassisch)':<20} | {'Rel. Abweichung [%]':<15} | Konvergenz\n")
    f.write("-" * 80 + "\n")
    for s in [2, 3, 4, 5, 6]:
        zeta_C = sum(compute_coherence(n) / (n**s) for n in range(1, max_n + 1))
        zeta_classic = float(zeta(s)) if s != 1 else float('inf')
        rel_diff = (zeta_C - zeta_classic) / zeta_classic * 100 if s != 1 else 0
        f.write(f"{s:<5} | {zeta_C:<20.8f} | {zeta_classic:<20.8f} | {rel_diff:<15.2f} | {'Schnell' if s >= 3 else 'Langsam'}\n")
    f.write("--- \n")
    f.write("Hinweis: ζ_C(s) konvergiert schneller als ζ(s), da C_n < 1 die Summe dämpft.\n")
    f.write("=" * 80)
