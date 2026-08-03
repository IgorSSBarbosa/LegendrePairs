import sys
import math
from itertools import combinations_with_replacement

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def units_mod(n):
    """Return list of units modulo n (numbers coprime to n)."""
    return [x for x in range(1, n) if gcd(x, n) == 1]

def build_multiplication_table(units, n):
    """Return dict (a,b) -> (a*b) % n for a,b in units."""
    table = {}
    for a in units:
        for b in units:
            table[(a, b)] = (a * b) % n
    return table

def enumerate_characters(units, mult_table):
    """
    Enumerate all homomorphisms from the group of units to {±1}.
    Returns a list of dicts: {unit: sign}.
    """
    # units[0] should be 1
    if units[0] != 1:
        # sort so that 1 is first
        units.sort()
    # We'll use backtracking over the indices of units (skip 1)
    chars = []
    n = len(units)
    # map from unit to index
    idx = {u: i for i, u in enumerate(units)}
    
    def backtrack(pos, assignment):
        # assignment: list of length pos, assignment[0] = 1 for unit 1
        if pos == n:
            # all units assigned, check consistency for all products
            # but we already forced consistency during assignment
            chars.append(dict(zip(units, assignment)))
            return
        u = units[pos]
        # Check if u can be expressed as product of two already assigned elements
        forced = None
        for a in units[:pos]:
            for b in units[:pos]:
                if mult_table[(a, b)] == u:
                    # a and b have known signs
                    val = assignment[idx[a]] * assignment[idx[b]]
                    if forced is None:
                        forced = val
                    elif forced != val:
                        # inconsistency, this assignment impossible
                        return
        if forced is not None:
            assignment.append(forced)
            backtrack(pos + 1, assignment)
            assignment.pop()
        else:
            # try both possibilities
            for sign in (1, -1):
                assignment.append(sign)
                backtrack(pos + 1, assignment)
                assignment.pop()
    
    # start with 1 -> 1
    backtrack(1, [1])  # assignment for unit 1 is 1
    return chars

def generate_candidates_general(ell):
    """Generate all candidate sequences from characters and parameters."""
    if ell < 2:
        return []
    units = units_mod(ell)
    if not units:
        # only unit is 1? For ell=2, units=[1]
        pass
    # Special case: if ell=2, U={1} trivial, characters just {1:1}
    # Build multiplication table
    mult_table = build_multiplication_table(units, ell)
    chars = enumerate_characters(units, mult_table)
    candidates = []
    # Parameters: s = a0, t = global factor for units, c = non-unit value
    for s in (1, -1):
        for t in (1, -1):
            for c in (1, -1):
                for chi in chars:
                    a = [c] * ell  # default for non-units
                    a[0] = s
                    for u in units:
                        # u is a unit, value = t * chi[u]
                        a[u] = t * chi[u]
                    candidates.append(tuple(a))
    # Remove duplicates (there may be many)
    return list(set(candidates))

def periodic_autocorrelation(seq):
    """Compute PAF C(s) for s=0..n-1."""
    n = len(seq)
    c = [0] * n
    for s in range(n):
        total = 0
        for i in range(n):
            total += seq[i] * seq[(i + s) % n]
        c[s] = total
    return c

def is_legendre_pair(a, b):
    """Check if (a,b) is a Legendre pair."""
    n = len(a)
    if len(b) != n:
        return False
    ca = periodic_autocorrelation(a)
    cb = periodic_autocorrelation(b)
    for s in range(1, n):
        if ca[s] + cb[s] != -2:
            return False
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python legendre_pair_search.py <ell>")
        sys.exit(1)
    try:
        ell = int(sys.argv[1])
    except ValueError:
        print("Please provide an integer.")
        sys.exit(1)

    if ell < 2:
        print("ell must be >= 2.")
        sys.exit(1)

    print(f"Testing ell = {ell}")
    candidates = generate_candidates_general(ell)
    if not candidates:
        print("No candidates generated (maybe ell too small?).")
        return

    print(f"Generated {len(candidates)} candidate sequences.")

    # Cache PAFs
    paf_cache = {}
    for seq in candidates:
        paf_cache[seq] = periodic_autocorrelation(seq)

    found_pairs = []
    # Check all unordered pairs (including a==b)
    for i, a in enumerate(candidates):
        for b in candidates[i:]:
            if is_legendre_pair(a, b):
                found_pairs.append((a, b))

    if not found_pairs:
        print("No Legendre pairs found.")
        return

    # Save to file
    filename = f"LP_{ell}.txt"
    with open(filename, "w") as f:
        f.write(f"Legendre pairs for ell = {ell}\n")
        f.write(f"Total pairs found: {len(found_pairs)}\n\n")
        for idx, (a, b) in enumerate(found_pairs, 1):
            f.write(f"Pair {idx}:\n")
            f.write(f"  a = {a}\n")
            f.write(f"  b = {b}\n")
            ca = paf_cache[a]
            cb = paf_cache[b]
            f.write("  s : C_a(s) + C_b(s)\n")
            for s in range(1, ell):
                f.write(f"  {s:2d} : {ca[s] + cb[s]:3d}\n")
            f.write("\n")

    print(f"Saved {len(found_pairs)} Legendre pair(s) to file '{filename}'.")

if __name__ == "__main__":
    main()