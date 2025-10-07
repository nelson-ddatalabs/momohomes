# COMPLETE ALGORITHM SPECIFICATION
## Gap Redistribution C-Channel Optimizer

---

## CONFIRMED REQUIREMENTS

### User Answers:
1. **Multiple C-channels (if gap > 18")**: Side-by-side vertical strips
2. **Split orientation**:
   - C) Depends on gap geometry (longest dimension)
   - D) Minimize total cassette movement distance
3. **Shifting strategy**: C) Optimize to minimize total movement distance
4. **Gap minimization**: Gap should be MINIMUM, C-channel plugs the gap
5. **Gap < 1.5"**: Adjust cassette size to reach 100% coverage
6. **Bounds validation**: Try different split location OR different strategy
   - **NEVER** extend polygon
   - **NEVER** be below or above 100% coverage
   - **ACCURACY AND PRECISENESS IS PARAMOUNT**

---

## CRITICAL INSIGHTS FROM ANSWER #5

### **MAJOR CLARIFICATION NEEDED**

User said: "If gap is smaller than c-channel minimum, **adjust cassette size** to make sure we reach 100% coverage"

This introduces a **FUNDAMENTAL CHANGE**:
- Previously: Standard sizes only (8x6, 8x4, 6x8)
- Now: **Custom cassette sizes allowed** to achieve 100% coverage

### **IMPLICATIONS:**

If we can adjust cassette sizes, then:
1. We can ELIMINATE boundary gaps entirely by resizing boundary cassettes
2. We DON'T NEED to shift cassettes to create center gaps
3. We DON'T NEED C-channels at all (just resize cassettes to fill 100%)

### **THE CONTRADICTION:**

- **Requirement**: Use C-channels to fill gaps
- **But**: If we can resize cassettes, we can eliminate all gaps
- **Question**: When/why do we use C-channels vs resize cassettes?

---

## POSSIBLE INTERPRETATIONS

### **INTERPRETATION A: Minimize Custom Sizes**
Try to use standard sizes as much as possible, only resize when necessary:
1. Place standard cassettes touching
2. Detect boundary gaps
3. **IF** gap >= 1.5": Use C-channel redistribution (shift to center)
4. **IF** gap < 1.5": Resize boundary cassettes to eliminate gap

**Problem**: Still creates boundary gaps >= 1.5", contradicting "no boundary gaps"

---

### **INTERPRETATION B: Resize to Eliminate ALL Boundary Gaps**
Always resize boundary cassettes to achieve perfect fit:
1. Place cassettes greedily
2. Detect boundary gaps (any size)
3. Resize boundary cassettes to eliminate gaps
4. Result: 100% coverage with cassettes only, no C-channels needed

**Problem**: No C-channels are ever used, contradicting the whole system purpose

---

### **INTERPRETATION C: Hybrid Approach**
Use C-channels for gaps within limits, resize for gaps outside limits:
1. Place standard cassettes touching
2. Detect boundary gaps
3. **IF** 1.5" <= gap <= 18": Shift cassettes + C-channel
4. **IF** gap < 1.5": Resize cassettes to eliminate gap
5. **IF** gap > 18": ??? (Multiple C-channels? Resize?)

---

### **INTERPRETATION D: Always Use C-Channels, Resize as Needed**
The goal is to ALWAYS have C-channels, adjust cassettes to make it work:
1. Place cassettes touching
2. Detect boundary gaps
3. **IF** gap < 1.5": ENLARGE gap by shrinking cassettes until gap >= 1.5"
4. Shift cassettes to move gap to center
5. Use C-channel to fill center gap
6. Result: Always have C-channels, even if we had to create bigger gap

---

## CRITICAL QUESTIONS FOR USER

### **Q1: What is the PRIMARY GOAL?**
   - **A)** Use standard cassette sizes ONLY (never resize)
   - **B)** ALWAYS use C-channels (resize cassettes if needed to make C-channels work)
   - **C)** Achieve 100% coverage (use C-channels when possible, resize when necessary)
   - **D)** Something else?

---

### **Q2: When should we resize cassettes vs use C-channels?**

**Scenario 1:** Boundary gap = 18" (exactly max C-channel width)
   - **A)** Resize boundary cassette to eliminate gap (no C-channel)
   - **B)** Shift cassettes + use 18" C-channel in center

**Scenario 2:** Boundary gap = 12" (within C-channel range)
   - **A)** Resize boundary cassette to eliminate gap (no C-channel)
   - **B)** Shift cassettes + use 12" C-channel in center

**Scenario 3:** Boundary gap = 0.5" (below C-channel min)
   - **A)** Resize boundary cassette by 0.5" to eliminate gap
   - **B)** Shrink cassettes to ENLARGE gap to 1.5", then use C-channel
   - **C)** Accept 99.95% coverage (leave tiny gap)

**Scenario 4:** Boundary gap = 24" (above C-channel max)
   - **A)** Resize boundary cassette to eliminate gap (no C-channel)
   - **B)** Use 2 × 12" C-channels side-by-side in center
   - **C)** Use 1 × 18" + 1 × 6" C-channels in center
   - **D)** Other approach?

---

### **Q3: Are C-channels REQUIRED or OPTIONAL?**
   - **A)** REQUIRED: Every floor plan must have C-channels (resize cassettes if needed to create gaps for C-channels)
   - **B)** OPTIONAL: Use C-channels when gaps exist within 1.5"-18" range, otherwise resize cassettes
   - **C)** PREFERRED: Prefer C-channels, but resize if C-channels can't work

---

### **Q4: What cassette sizes are acceptable?**

**Current standard sizes:**
- 8x6, 8x4, 8x2
- 6x8, 6x4, 6x2
- 4x8, 4x6, 4x4, 4x2
- 2x8, 2x6, 2x4, 2x2

**If we can resize, can we use:**
- **A)** Any integer dimensions (7x6, 9x4, 11x5, etc.)
- **B)** Fractional dimensions (7.5x6, 8.3x4, etc.)
- **C)** Must stay at standard sizes (no custom sizes after all)
- **D)** Only resize in specific increments (e.g., 0.5' increments)

---

### **Q5: Cassette resizing rules:**
   - **A)** Can resize ANY cassette to any dimension
   - **B)** Can only resize BOUNDARY cassettes (touching polygon edge)
   - **C)** Can resize but prefer keeping standard sizes when possible
   - **D)** Other constraints?

---

### **Q6: The 100% COVERAGE requirement - clarify:**

You said: "NEVER BE BELOW OR MORE THAN 100% COVERAGE"

Does this mean:
   - **A)** EXACTLY 100.00% (no tolerance, perfect precision)
   - **B)** Within 99.99% - 100.01% (tiny tolerance for floating point)
   - **C)** Within 99.9% - 100.1% (small tolerance)
   - **D)** Other tolerance?

---

## EXAMPLE SCENARIOS TO CLARIFY

### **Example 1: Umbra XL Current State**
- Polygon: 1080 sq ft
- Standard cassettes: 1056 sq ft
- Boundary gap: 24 sq ft (1.5' × 16' at right edge)

**Option A: C-channel approach**
1. Shift cassettes to create 1.5' × 16' center gap
2. Insert 18" × 16' C-channel
3. Result: 1056 (cassettes) + 24 (C-channel) = 1080 sq ft ✓

**Option B: Resize approach**
1. Identify rightmost cassettes (8x6 each)
2. Resize last column to 8.25x6 (add 0.25' width each)
3. 6 cassettes × 0.25' × 6' = 9 sq ft gained
4. Still 15 sq ft short... need more resizing
5. Resize multiple cassettes to gain 24 sq ft total
6. Result: 1080 sq ft cassettes, 0 sq ft C-channel ✓

**Which approach should we use?**

---

### **Example 2: Tiny Gap**
- Polygon: 500 sq ft
- Standard cassettes: 499.5 sq ft
- Boundary gap: 0.5 sq ft (0.5' × 1' = 6" × 12")

**Option A: Resize cassette**
1. Resize one boundary cassette slightly
2. Result: 500 sq ft cassettes, no C-channel

**Option B: Force C-channel**
1. Shrink cassettes to create 1.5" gap minimum
2. Shift to center
3. Use 1.5" C-channel
4. Result: 498.75 cassettes + 1.25 C-channel = 500 ✓

**Which approach should we use?**

---

### **Example 3: Large Gap**
- Polygon: 1000 sq ft
- Standard cassettes: 950 sq ft
- Boundary gap: 50 sq ft (2.5' × 20')

**Option A: Multiple C-channels**
1. 2.5' = 30" (exceeds 18" max)
2. Use 2 × 15" C-channels side-by-side (total 30")
3. Shift cassettes to create 30" × 20' center gap
4. Result: 950 + 50 = 1000 ✓

**Option B: Resize cassettes**
1. Enlarge boundary cassettes by 50 sq ft total
2. Result: 1000 sq ft cassettes, no C-channel

**Which approach should we use?**

---

## WHAT I THINK YOU MEAN (Best Guess)

Based on all requirements, I believe:

### **PRIMARY GOAL:**
- Achieve EXACTLY 100% coverage (paramount requirement)
- Use C-channels as the PRIMARY gap-filling method
- Only resize cassettes when C-channels cannot work

### **DECISION TREE:**

```
1. Place standard cassettes touching
2. Detect boundary gap

3. IF gap == 0:
     → Done! 100% coverage achieved

4. IF 1.5" <= gap_width <= 18":
     → Use C-channel approach (shift + fill)

5. IF gap_width > 18":
     → Use multiple C-channels (side-by-side)

6. IF gap_width < 1.5":
     → Resize boundary cassette to eliminate gap
     → (C-channel too small, must resize)

7. IF gap area too complex for C-channels:
     → Resize cassettes as needed
     → Ensure 100% coverage
```

### **Cassette Sizing:**
- Prefer standard sizes (8x6, 8x4, etc.)
- Allow fractional sizes ONLY when necessary (7.5x6, 8.25x4)
- Only resize boundary cassettes (not interior ones)

### **Coverage:**
- Target: EXACTLY 100.00%
- Tolerance: ±0.01% (floating point precision)
- Never extend beyond polygon
- Never leave any gaps unfilled

---

## IS THIS CORRECT?

Please confirm or correct my understanding, especially:
1. When to use C-channels vs resize cassettes
2. What cassette sizes are allowed
3. The exact coverage tolerance
4. Priority: C-channels first, resize only when necessary?

Once confirmed, I can implement the complete algorithm!
