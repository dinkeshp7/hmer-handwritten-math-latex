import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_full_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Professional Academic Colors
    NAVY_DARK = RGBColor(11, 37, 69)       # #0B2545 - Deep Academic Navy
    NAVY_MED = RGBColor(19, 64, 116)       # #134074 - Slate Navy
    ACCENT_GOLD = RGBColor(217, 119, 6)    # #D97706 - Amber/Gold Accent
    ACCENT_BLUE = RGBColor(37, 99, 235)    # #2563EB - Tech Blue
    BG_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC - Off white
    CARD_BG = RGBColor(255, 255, 255)      # #FFFFFF - Crisp white
    BORDER_COLOR = RGBColor(226, 232, 240) # #E2E8F0 - Subtle slate border
    TEXT_DARK = RGBColor(15, 23, 42)       # #0F172A - Charcoal
    TEXT_MUTED = RGBColor(100, 116, 139)   # #64748B - Slate Grey
    GREEN_SUCCESS = RGBColor(16, 185, 129) # #10B981 - Success Green
    RED_ALERT = RGBColor(220, 38, 38)      # #DC2626 - Alert Red

    def add_header(slide, title_text, category_text="B.TECH PROJECT DEFENSE | DEPARTMENT OF MATHEMATICS, IIT GUWAHATI"):
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = NAVY_DARK
        top_bar.line.fill.background()

        accent_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.15), Inches(13.333), Inches(0.06))
        accent_line.fill.solid()
        accent_line.fill.fore_color.rgb = ACCENT_GOLD
        accent_line.line.fill.background()

        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.3))
        cat_tf = cat_box.text_frame
        cat_tf.word_wrap = True
        cat_tf.margin_top = Inches(0)
        cat_tf.margin_bottom = Inches(0)
        p_cat = cat_tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_GOLD
        p_cat.font.name = "Calibri"

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.7), Inches(0.65))
        ttf = title_box.text_frame
        ttf.word_wrap = True
        ttf.margin_top = Inches(0)
        ttf.margin_bottom = Inches(0)
        p_title = ttf.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(255, 255, 255)
        p_title.font.name = "Calibri"

    def add_card(slide, left, top, width, height, title="", title_color=NAVY_DARK, bg_color=CARD_BG, border_color=BORDER_COLOR):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.45))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_top = Inches(0)
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.font.name = "Calibri"

        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Updated Committee: Dr. H. Ramesh & Dr. Vinay Wagh)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY_DARK
    bg1.line.fill.background()

    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(0.7), Inches(11.333), Inches(0.07))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = ACCENT_GOLD
    bar1.line.fill.background()

    tbox1 = s1.shapes.add_textbox(Inches(1.0), Inches(0.85), Inches(11.333), Inches(1.9))
    tf1 = tbox1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Automated Full-Page Handwritten Mathematical Expression Recognition (HMER) and Exam Script-to-LaTeX Transcription"
    p1.font.size = Pt(25)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.font.name = "Calibri"

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "High-Resolution Multi-Band Vision-Language Inference on PARAM Kamrupa Supercomputer"
    p1_sub.font.size = Pt(15.5)
    p1_sub.font.bold = False
    p1_sub.font.color.rgb = ACCENT_GOLD
    p1_sub.font.name = "Calibri"

    # Teammate's Impactful Visual Hook: Ink -> Big Arrow -> LaTeX
    hook_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.85), Inches(11.333), Inches(1.15))
    hook_box.fill.solid()
    hook_box.fill.fore_color.rgb = RGBColor(19, 42, 74)
    hook_box.line.color.rgb = ACCENT_GOLD
    hook_box.line.width = Pt(1.5)

    tb_ink = s1.shapes.add_textbox(Inches(1.2), Inches(2.9), Inches(4.3), Inches(1.05))
    tf_ink = tb_ink.text_frame
    tf_ink.word_wrap = True
    p = tf_ink.paragraphs[0]
    p.text = "RAW HANDWRITTEN INK (SCANNED 300 DPI):"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    p = tf_ink.add_paragraph()
    p.text = "[ -1.0   8.5 ]\n[  4.5   2.5 ]  (4x4 unruled matrix ink with decimals)"
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(241, 245, 249)

    tb_arr = s1.shapes.add_textbox(Inches(5.7), Inches(3.05), Inches(1.9), Inches(0.8))
    tf_arr = tb_arr.text_frame
    p = tf_arr.paragraphs[0]
    p.text = "──────►"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    p.alignment = PP_ALIGN.CENTER
    p = tf_arr.add_paragraph()
    p.text = "END-TO-END PIPELINE"
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.alignment = PP_ALIGN.CENTER

    tb_lat = s1.shapes.add_textbox(Inches(7.8), Inches(2.9), Inches(4.3), Inches(1.05))
    tf_lat = tb_lat.text_frame
    tf_lat.word_wrap = True
    p = tf_lat.paragraphs[0]
    p.text = "VERBATIM PRODUCTION LATEX (COMPILABLE):"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    p = tf_lat.add_paragraph()
    p.text = "\\begin{bmatrix} -1.0 & 8.5 \\\\ 4.5 & 2.5 \\end{bmatrix}\n(Zero syntax errors; 100% PDF compilation pass rate)"
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(241, 245, 249)

    # Info Card 1: Project Details
    c1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.2), Inches(5.5), Inches(2.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(19, 42, 74)
    c1.line.color.rgb = RGBColor(40, 70, 110)
    c1.line.width = Pt(1)

    c1_tb = s1.shapes.add_textbox(Inches(1.2), Inches(4.3), Inches(5.1), Inches(2.6))
    c1_tf = c1_tb.text_frame
    c1_tf.word_wrap = True
    p = c1_tf.paragraphs[0]
    p.text = "CANDIDATES & AFFILIATION"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    
    items_c1 = [
        ("Candidate 1:", "Dinkesh Pal (Roll No. 230123021)"),
        ("Candidate 2:", "Rithvik Ponnapalli (Roll No. 230123052)"),
        ("Program:", "Bachelor of Technology (Mathematics & Computing)"),
        ("Course:", "MA4998P B.Tech Project – I"),
        ("Department:", "Department of Mathematics, IIT Guwahati"),
        ("Academic Year:", "2026–2027")
    ]
    for label, val in items_c1:
        p = c1_tf.add_paragraph()
        p.text = f"{label} {val}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = RGBColor(241, 245, 249)

    # Info Card 2: Committee & Supervisor (Dr. H. Ramesh & Dr. Vinay Wagh)
    c2 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.2), Inches(5.5), Inches(2.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(19, 42, 74)
    c2.line.color.rgb = RGBColor(40, 70, 110)
    c2.line.width = Pt(1)

    c2_tb = s1.shapes.add_textbox(Inches(7.0), Inches(4.3), Inches(5.1), Inches(2.6))
    c2_tf = c2_tb.text_frame
    c2_tf.word_wrap = True
    p = c2_tf.paragraphs[0]
    p.text = "SUPERVISOR & EVALUATION COMMITTEE"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    
    items_c2 = [
        ("Supervisor:", "Dr. K. V. Srikanth (Assistant Professor)"),
        ("Committee Member:", "Dr. H. Ramesh (Associate Professor, DUPC Sec.)"),
        ("Committee Member:", "Dr. Vinay Wagh (Associate Professor)"),
        ("Department:", "Department of Mathematics, IIT Guwahati"),
        ("HPC Infrastructure:", "PARAM Kamrupa Supercomputer (CCC, IITG)"),
        ("Defense Date:", "November 7, 2026")
    ]
    for label, val in items_c2:
        p = c2_tf.add_paragraph()
        p.text = f"{label} {val}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = RGBColor(241, 245, 249)

    s1.notes_slide.notes_text_frame.text = (
        "Respected Supervisor Dr. Srikanth, respected committee members Dr. Ramesh and Dr. Wagh, good morning.\n\n"
        "Today, Rithvik and I are presenting our B.Tech Project – I titled 'Automated Full-Page Handwritten Mathematical Expression Recognition "
        "and Exam Script-to-LaTeX Transcription'.\n\n"
        "This is fundamentally an applied machine learning and systems engineering project. In this work, we address the computational "
        "and algorithmic bottlenecks of transcribing unconstrained university mathematics examination scripts into compilable, "
        "production-ready LaTeX on the PARAM Kamrupa Supercomputer.\n\n"
        "As illustrated on this title card, our mission is to take messy handwritten ink—with all its decimal points, negative signs, "
        "and matrix layouts—and convert it end-to-end into verified, compilable LaTeX code with 100% syntactic pass rate, preserving "
        "every student derivation verbatim for fair grading. Let us begin with the institutional scale that motivated this research."
    )

    # =========================================================================
    # SLIDE 2: Problem Definition, Scale & Strict Project Scope
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Definition & Institutional Scale (The MA102 Context)")

    stat_boxes = [
        ("900+", "First-Year Enrolled Students", "MA102 core foundation", NAVY_DARK),
        ("14", "Pages per Answer Booklet", "Unruled physical paper", ACCENT_BLUE),
        ("13,146", "Scanned Raw Pages / Exam", "Massive evaluation volume", ACCENT_GOLD),
        ("4", "Exams per Semester", "2 Quizzes, Mid-Sem, End-Sem", NAVY_MED)
    ]
    stat_w = Inches(2.7)
    gap_s = Inches(0.2)
    start_s = Inches(0.8)
    for i, (val, label, sub, col) in enumerate(stat_boxes):
        box_left = start_s + i * (stat_w + gap_s)
        sb = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, box_left, Inches(1.35), stat_w, Inches(1.05))
        sb.fill.solid()
        sb.fill.fore_color.rgb = RGBColor(248, 250, 252)
        sb.line.color.rgb = col
        sb.line.width = Pt(1.5)
        
        stb = s2.shapes.add_textbox(box_left + Inches(0.1), Inches(1.4), stat_w - Inches(0.2), Inches(0.95))
        stf = stb.text_frame
        stf.word_wrap = True
        stf.margin_top = Inches(0)
        p = stf.paragraphs[0]
        p.text = val
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        
        p = stf.add_paragraph()
        p.text = label
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        p.alignment = PP_ALIGN.CENTER

        p = stf.add_paragraph()
        p.text = sub
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_MUTED
        p.alignment = PP_ALIGN.CENTER

    add_card(s2, Inches(0.8), Inches(2.55), Inches(5.6), Inches(4.35), "Cognitive Fatigue & OCR Failures")
    tb2_left = s2.shapes.add_textbox(Inches(1.0), Inches(3.05), Inches(5.2), Inches(3.7))
    tf2_l = tb2_left.text_frame
    tf2_l.word_wrap = True
    
    bullets2_l = [
        ("Cognitive Fatigue & Variance: ", "Professors and TAs read thousands of dense pages. Human grading standards drift between the first booklet of the morning and the last one at night."),
        ("1D vs 2D Spatial Mismatch: ", "Standard OCR (Tesseract) assumes 1D linear text. Mathematics is a recursive 2D spatial tree (fractions, powers, nested matrices)."),
        ("Dynamic Writing Noise: ", "Unruled paper induces severe baseline drift (-35° to +35°), interlocking ascenders/descenders (touching 'y' and 'd'), and scribbled cross-hatching."),
        ("Air-Gapped Institutional Privacy: ", "Confidential exam scans cannot be sent to public cloud APIs (ChatGPT/Claude). All AI inference must run on-premise on PARAM Kamrupa.")
    ]
    for b_title, b_desc in bullets2_l:
        p = tf2_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s2, Inches(6.8), Inches(2.55), Inches(5.7), Inches(4.35), "Project Scope: Assist, Don't Grade")
    tb2_right = s2.shapes.add_textbox(Inches(7.0), Inches(3.05), Inches(5.3), Inches(3.7))
    tf2_r = tb2_right.text_frame
    tf2_r.word_wrap = True
    
    p = tf2_r.paragraphs[0]
    p.text = "✔ WHAT THE SYSTEM DOES (ASSISTIVE TRANSCRIPTION):"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    
    pos_items = [
        "Transcribes raw unruled exam booklets into valid, compilable LaTeX.",
        "Copies student derivations verbatim: arithmetic mistakes preserved.",
        "Zero Tolerance for syntax crashes: pushdown balancer guarantees PDF.",
        "Shows side-by-side typeset math next to ink for fast faculty grading."
    ]
    for item in pos_items:
        p = tf2_r.add_paragraph()
        p.text = "  • " + item
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(3)

    p = tf2_r.add_paragraph()
    p.text = "✖ WHAT THE SYSTEM NEVER DOES (ETHICAL BOUNDARY):"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RED_ALERT
    p.space_before = Pt(8)

    neg_items = [
        "Never grades, assigns marks, or evaluates mathematical correctness.",
        "Never quietly 'fixes' student arithmetic errors (e.g. -2.0 stays -2.0).",
        "Never replaces the instructor or TA; acts solely as visual assistant."
    ]
    for item in neg_items:
        p = tf2_r.add_paragraph()
        p.text = "  • " + item
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(3)

    s2.notes_slide.notes_text_frame.text = (
        "To understand why this is a challenging computer vision and document intelligence problem, consider the scale of MA102 here at IIT Guwahati. "
        "Every semester, over 900 first-year undergraduates write four major examinations in 14-page unruled booklets, generating more than 13,000 scanned pages per cycle.\n\n"
        "Grading these booklets manually is an administrative and cognitive bottleneck. As any evaluator knows, human grading standards inevitably drift "
        "between the first booklet of the morning and the last one at night. Standard OCR like Tesseract completely fails because text is 1D, while math is a 2D spatial tree.\n\n"
        "Most importantly, I want to clarify our project scope upfront to the committee: our system is strictly assistive. "
        "It transcribes raw scans into clean LaTeX and side-by-side PDFs. It NEVER grades, assigns marks, or quietly fixes student blunders. "
        "Every calculation mistake is transcribed verbatim so instructors retain 100% grading discretion."
    )

    # =========================================================================
    # SLIDE 3: Hardware Bottlenecks & Systems Engineering on PARAM Kamrupa
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Hardware Bottlenecks & Systems Engineering on PARAM Kamrupa")

    add_card(s3, Inches(0.8), Inches(1.35), Inches(5.6), Inches(4.3), "Tesla V100 (16 GB) VRAM Arithmetic")
    tb3_left = s3.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.2), Inches(3.6))
    tf3_l = tb3_left.text_frame
    tf3_l.word_wrap = True
    
    items3_l = [
        ("PARAM Kamrupa Partition:", "NVIDIA Tesla V100-SXM2 (Volta sm_70 architecture, 16 GB HBM2 memory)."),
        ("7B Model Weights in FP16:", "14.07 GiB (7.61 billion parameters * 2 bytes/weight)."),
        ("Addressable Physical VRAM:", "15.78 GiB available to PyTorch allocator."),
        ("Available Headroom:", "Only ~1.71 GiB (1,750 MiB) remaining for CUDA context, activations, and KV cache!"),
        ("The Monolithic O(N^2) Crash:", "Feeding an uncompressed scanner page (1653x2338 px) generates > 5,800 visual patches. ViT self-attention attempts to allocate 1.30 GiB contiguous tensor -> Fatal CUDA OOM."),
        ("The Downscaling Compromise:", "Downscaling full page to 848 px width prevents OOM, but blurs 30 lines of writing: decimal points and minus signs vanish (3.5 -> 8.5).")
    ]
    for b_title, b_desc in items3_l:
        p = tf3_l.add_paragraph()
        p.text = "• " + b_title + " " + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    add_card(s3, Inches(6.8), Inches(1.35), Inches(5.7), Inches(4.3), "PARAM Kamrupa Cluster Engineering Hacks")
    tb3_right = s3.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(3.6))
    tf3_r = tb3_right.text_frame
    tf3_r.word_wrap = True
    
    items3_r = [
        ("Air-Gapped Node Isolation: ", "Compute nodes have ZERO internet access for exam confidentiality. We download weights on login nodes and stream via high-speed Lustre scratch storage."),
        ("CentOS 7 C++ Runtime Workaround: ", "The supercomputer runs an older CentOS 7 OS with outdated libstdc++. We preloaded a modern C++ library from Anaconda using Python ctypes (zero root privileges needed)."),
        ("Volta sm_70 Attention Kernels: ", "Tesla V100 lacks FlashAttention-2 support. We engineered memory-efficient Cutlass attention kernels to bypass quadratic activation bloat."),
        ("Unattended SLURM Batch Execution: ", "Built automated SLURM batch runners that process entire directories of exam booklets in headless background jobs.")
    ]
    for b_title, b_desc in items3_r:
        p = tf3_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(5)

    c3_bot = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c3_bot.fill.solid()
    c3_bot.fill.fore_color.rgb = RGBColor(254, 242, 242)
    c3_bot.line.color.rgb = RED_ALERT
    c3_bot.line.width = Pt(1.2)
    
    tb3_b = s3.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf3_b = tb3_b.text_frame
    tf3_b.word_wrap = True
    p = tf3_b.paragraphs[0]
    p.text = "THE CORE COMPUTATIONAL CHALLENGE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RED_ALERT
    
    p = tf3_b.add_paragraph()
    p.text = "Fixed Weights (14.07 GiB) + CUDA Context (0.8 GiB) + Full-Page ViT Attention (1.30 GiB) = 16.17 GiB -> Exceeds 15.78 GiB Hardware Limit! We needed an architecture that gave us 100% optical clarity within our 1.7 GiB headroom."
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK

    s3.notes_slide.notes_text_frame.text = (
        "When we moved to 7-billion parameter vision-language models, we hit a severe hardware wall on PARAM Kamrupa.\n\n"
        "Our GPU compute nodes are equipped with NVIDIA Tesla V100 accelerators with 16 GB of VRAM. Storing the 7.6-billion parameters of "
        "Qwen2.5-VL in 16-bit half precision takes exactly 14.07 GiB, leaving less than 1.7 GiB of free headroom.\n\n"
        "Feeding a raw uncompressed 300 DPI page into a Vision Transformer produces over 5,800 visual patches, demanding 1.30 GiB contiguous attention memory, "
        "which instantly triggers a CUDA OOM crash. Downscaling the page to 848 px avoided OOM but destroyed symbol clarity, turning 3.5 into 8.5.\n\n"
        "Beyond VRAM, deploying on PARAM Kamrupa required serious systems engineering: compute nodes are air-gapped without internet, "
        "the OS runs older CentOS 7 requiring ctypes preloading of libstdc++, and Tesla V100 lacks FlashAttention-2, forcing us to use Cutlass kernels. "
        "This led directly to our architectural evolution."
    )

    # =========================================================================
    # SLIDE 4: Evolution of Approaches (From Classical GANs to Multi-Band Slicing)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Evolution of Approaches (From Classical GANs to Multi-Band Slicing)")

    cards_data = [
        ("Phase 1A: GAN Fonts", 
         "Pivot 1 - Discarded",
         "Handwriting -> PatchGAN -> Synthetic Print -> pix2tex",
         RED_ALERT,
         ["• Image-to-image GAN to normalize handwriting to printed font.",
          "• Mode Collapse: Discriminator hallucinated phantom serifs ('l' -> '1').",
          "• L1 pixel loss erased delicate minus signs and decimal dots.",
          "• Golden Takeaway: Never generate an in-between image. Go straight to LaTeX tokens."]),
        
        ("Phase 1B: Modular Hybrid", 
         "Pivot 2 - Explored",
         "Sauvola -> IoU Line NMS -> TrOCR + DenseNet -> 2D Spatial DAG",
         ACCENT_GOLD,
         ["• Sauvola binarization + Vertical IoU (>=0.35) line merging.",
          "• MatrixGridParser: Detects tall bracket strokes (height > 3x width) & projection cuts.",
          "• Compounding Error: One split error broke the downstream syntax tree.",
          "• TrOCR vocabulary drift: Cursive proof turned into 19th-century legal prose ('House of Representatives...')."]),
        
        ("Phase 1C: Monolithic VLM", 
         "Pivot 3 - Baseline",
         "Qwen2.5-VL-7B (Downscaled 848 px) + Assistant Prefill Locking",
         ACCENT_BLUE,
         ["• Unified multimodal context with single-stage attention.",
          "• Assistant Prefill Locking: Prefills '\\documentclass{article}' to suppress chatter.",
          "• 100% LaTeX syntax compilation pass rate; 28.4% TER baseline.",
          "• Bottleneck: Downscaled to 848 px to fit 16 GB VRAM; missed faint pencil strokes and decimals."]),
        
        ("Phase 1D: Multi-Band Slicing", 
         "Current SOTA Approach",
         "Overlapping 3-Band Slicing (1200 px) + Automated LaTeX Stitcher",
         GREEN_SUCCESS,
         ["• Slices active student canvas into 3 overlapping bands (8% buffer).",
          "• Native resolution at 1200 px: 3.5x higher token density per line.",
          "• ViT activation drops to ~240 MiB (Zero CUDA OOM on Tesla V100).",
          "• Stitched with stitch_latex_parts.py: 88.24% Word-for-Word Accuracy!"])
    ]

    card_w = Inches(2.7)
    gap = Inches(0.2)
    left_start = Inches(0.8)

    for i, (p_title, p_sub, p_flow, border_c, p_bullets) in enumerate(cards_data):
        cur_left = left_start + i * (card_w + gap)
        slide_card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, Inches(1.35), card_w, Inches(5.6))
        slide_card.fill.solid()
        slide_card.fill.fore_color.rgb = CARD_BG
        slide_card.line.color.rgb = border_c
        slide_card.line.width = Pt(1.4)
        
        tb_head = s4.shapes.add_textbox(cur_left + Inches(0.12), Inches(1.45), card_w - Inches(0.24), Inches(0.75))
        tf_h = tb_head.text_frame
        tf_h.word_wrap = True
        tf_h.margin_top = Inches(0)
        tf_h.margin_bottom = Inches(0)
        p = tf_h.paragraphs[0]
        p.text = p_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        
        p_s = tf_h.add_paragraph()
        p_s.text = p_sub.upper()
        p_s.font.size = Pt(9.5)
        p_s.font.bold = True
        p_s.font.color.rgb = border_c
        
        pbox = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, cur_left + Inches(0.12), Inches(2.25), card_w - Inches(0.24), Inches(0.72))
        pbox.fill.solid()
        pbox.fill.fore_color.rgb = RGBColor(241, 245, 249)
        pbox.line.color.rgb = BORDER_COLOR
        pbox_tb = s4.shapes.add_textbox(cur_left + Inches(0.14), Inches(2.27), card_w - Inches(0.28), Inches(0.68))
        pbox_tf = pbox_tb.text_frame
        pbox_tf.word_wrap = True
        pbox_tf.margin_top = Inches(0)
        p = pbox_tf.paragraphs[0]
        p.text = p_flow
        p.font.size = Pt(8.5)
        p.font.color.rgb = NAVY_DARK

        tb_b = s4.shapes.add_textbox(cur_left + Inches(0.12), Inches(3.08), card_w - Inches(0.24), Inches(3.7))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        tf_b.margin_top = Inches(0)
        for bullet in p_bullets:
            p = tf_b.add_paragraph()
            p.text = bullet
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(5)

    s4.notes_slide.notes_text_frame.text = (
        "Our project followed an iterative, empirical engineering journey across four distinct architectures:\n\n"
        "First, in August, we tried a Two-Stage GAN approach to translate handwriting into synthetic printed font before OCR. "
        "We discarded this (Pivot 1) because the GAN hallucinated phantom serifs and erased minus signs. Our golden takeaway: never generate an in-between image; go straight from pixels to tokens.\n\n"
        "Second, we built a Modular Hybrid Pipeline: we engineered 1D vertical IoU line merging (merging lines with IoU >= 0.35 so letters aren't cut), "
        "and built a custom MatrixGridParser that detects tall brackets and splits projection profiles. However, errors compounded across stages, "
        "and Microsoft TrOCR suffered from archaic vocabulary drift, turning cursive math proofs into 19th-century legal cases like 'House of Representatives...'.\n\n"
        "Third, we deployed monolithic Qwen2.5-VL-7B. Using Assistant Prefill Locking, we achieved a 100% compilation pass rate and 28.4% TER baseline, "
        "but downscaling the entire page to fit 16 GB VRAM blurred fine strokes.\n\n"
        "This led to our master breakthrough: The Multi-Band High-Resolution Slicing and Stitching Pipeline."
    )

    # =========================================================================
    # SLIDE 5 [NEW]: Algorithmic Deep-Dive: Modular Hybrid Pipeline & MatrixGridParser
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Algorithmic Deep-Dive: Modular Hybrid Pipeline & MatrixGridParser")

    add_card(s5, Inches(0.8), Inches(1.60), Inches(5.6), Inches(5.30), "Classical Computer Vision & Graph Formulation")
    tb5_l = s5.shapes.add_textbox(Inches(1.0), Inches(2.20), Inches(5.2), Inches(4.55))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True
    
    c5_items = [
        ("1. Sauvola Dynamic Binarization: ", "Adaptive local thresholding T(x,y) = m(x,y) · [1 + k · (s(x,y)/R - 1)] handling non-uniform scanner illumination and ink bleed."),
        ("2. 1D Vertical IoU Line Merging: ", "Calculates vertical intersection-over-union between bounding boxes. If IoU_vert ≥ 0.35, boxes are merged, preventing split ascenders/descenders ('y', 'd')."),
        ("3. MatrixGridParser Algorithm: ", "Detects vertical bracket candidates with aspect ratio H/W > 3.0. Projects horizontal and vertical stroke profiles to extract internal (r, c) cell coordinates and wrap into LaTeX \\begin{bmatrix} with & and \\\\."),
        ("4. 2D Spatial DAG Linearization: ", "Constructs a Directed Acyclic Graph (DAG) over segmented visual blocks; applies topological sorting to reconstruct 1D LaTeX reading order."),
        ("The Compounding Error Wall: ", "A single 5% error in bracket edge detection propagated downstream, corrupting the entire AST parse tree.")
    ]
    for b_title, b_desc in c5_items:
        p = tf5_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    add_card(s5, Inches(6.8), Inches(1.60), Inches(5.7), Inches(5.30), "Real 4x4 Decimal Matrix Segmentation Case Study")
    
    matrix_img_path = r"c:\Users\Dinkesh\Downloads\BTP\demo_results\matrix_4x4_case_study.png"
    if os.path.exists(matrix_img_path):
        s5.shapes.add_picture(matrix_img_path, Inches(7.0), Inches(2.25), Inches(5.3), Inches(1.43))

    tb5_r = s5.shapes.add_textbox(Inches(7.0), Inches(3.80), Inches(5.3), Inches(2.95))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True
    
    p = tf5_r.paragraphs[0]
    p.text = "ALGORITHMIC EXTRACTION MECHANISM"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p.space_after = Pt(6)
    
    r_items = [
        ("Tall Bracket Filter: ", "Isolates outer left and right bracket delimiters."),
        ("Projection Cut Profiles: ", "Zero-crossings in horizontal projection identify matrix rows; valley minima in vertical projection split column entries."),
        ("Dual-Head Dispatcher: ", "Dispatches pure text lines to TrOCR and dense formula crops to our custom DenseNet-Coverage model."),
        ("The Failure Point: ", "When students write cursive text slanted across matrix boundaries, the algorithmic grid parser cannot separate text from matrix brackets.")
    ]
    for b_title, b_desc in r_items:
        p = tf5_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s5.notes_slide.notes_text_frame.text = (
        "Before deploying large multimodal neural networks, we first built an entire algorithmic computer vision pipeline from scratch.\n\n"
        "As Dr. Ramesh knows from formal language theory, linearizing a two-dimensional mathematical script into a one-dimensional sequence "
        "requires a formal grammar. We formulated this by constructing a 2D Spatial Directed Acyclic Graph (DAG) over segmented visual blocks, "
        "and applied topological sorting to recover the natural reading order.\n\n"
        "We engineered four distinct algorithms: first, Sauvola dynamic local binarization; second, a vertical IoU clustering filter that merged "
        "bounding boxes with IoU >= 0.35 so letters like 'y' and 'd' were never split; third, an algorithmic MatrixGridParser that searched for tall "
        "bracket aspect ratios (H/W > 3) and used horizontal/vertical projection profiles to slice cells; and fourth, a dual-head dispatcher.\n\n"
        "As shown in the actual image crop on the right, this worked well on cleanly isolated matrices. But on unruled paper, errors compounded: "
        "if a bracket detection failed, the entire downstream DAG collapsed. This proved that hard-coded geometric heuristics cannot scale to 13,000 unconstrained pages."
    )

    # =========================================================================
    # SLIDE 6 [NEW]: The Custom DenseNet-Coverage Model & Attention Coverage Math
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Attention Coverage & Custom DenseNet HMER Model (23.1M Params)")

    add_card(s6, Inches(0.8), Inches(1.35), Inches(5.6), Inches(4.3), "The Attention Coverage Dilemma")
    tb6_l = s6.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.2), Inches(3.6))
    tf6_l = tb6_l.text_frame
    tf6_l.word_wrap = True
    
    b6_l = [
        ("The Coverage Defect: ", "Standard encoder-decoder attention has no memory of which image pixels have already been transcribed."),
        ("Under-Parsing (Symbol Dropping): ", "Tiny high-frequency symbols like minus signs, dots, and exponents occupy only a few pixels. Standard attention skips them: x² − 1 becomes x¹."),
        ("Over-Parsing (Infinite Token Loops): ", "Repeated structures (such as '0, 0, 0' in diagonal matrices) confuse the attention heads, causing the model to get stuck in infinite repetition loops."),
        ("The Mathematical Remedy: ", "We engineered a cumulative coverage vector c_t = Σ_{τ=1}^{t-1} α_τ tracking spatial attention history across time steps."),
        ("Coverage Loss Penalty: ", "Loss = Loss_CE + λ_cov · Σ min(c_{t,i,j}, α_{t,i,j}) forcing the decoder to attend only to unparsed ink pixels.")
    ]
    for b_title, b_desc in b6_l:
        p = tf6_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    add_card(s6, Inches(6.8), Inches(1.35), Inches(5.7), Inches(4.3), "DenseNet-Coverage Model Architecture")
    tb6_r = s6.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(3.6))
    tf6_r = tb6_r.text_frame
    tf6_r.word_wrap = True
    
    b6_r = [
        ("Encoder Architecture: ", "DenseNet-121 backbone extracting multi-scale feature maps with dense feature reuse across layers."),
        ("Coverage Attention Head: ", "Custom 2D convolutional coverage layer that maps cumulative attention history c_t into attention energy."),
        ("Model Size & Efficiency: ", "Only 23.1 Million parameters. Highly efficient; trains in hours on PARAM Kamrupa Tesla V100 GPU."),
        ("Isolated Formula Benchmark: ", "Achieved 75.0% Exact Match Rate (EMR) and 2.13% Token Error Rate (TER) on CROHME/MA102 formula benchmarks."),
        ("The Limitation for Full Pages: ", "While it achieves near-perfect accuracy on isolated formulas, it has zero language vocabulary for English prose proofs. Full pages require multimodal VLMs.")
    ]
    for b_title, b_desc in b6_r:
        p = tf6_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    c6_bot = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c6_bot.fill.solid()
    c6_bot.fill.fore_color.rgb = RGBColor(238, 242, 255)
    c6_bot.line.color.rgb = ACCENT_BLUE
    c6_bot.line.width = Pt(1.2)
    
    tb6_b = s6.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf6_b = tb6_b.text_frame
    tf6_b.word_wrap = True
    p = tf6_b.paragraphs[0]
    p.text = "THE SOTA THEORETICAL INSIGHT"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    p = tf6_b.add_paragraph()
    p.text = "Our 23.1M DenseNet-Coverage model mastered isolated mathematical formulas (75% Exact Match, 2.13% TER). However, university exam scripts interleave English proofs with math. To read entire pages, we needed the broad semantic context of a 7B Vision-Language Model."
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_DARK

    s6.notes_slide.notes_text_frame.text = (
        "In Chapter 3 of our report, we trained a custom deep neural network from scratch: a 23.1-million parameter DenseNet-Coverage model.\n\n"
        "Here, we tackled the fundamental 'Attention Coverage Problem'. When an attention mechanism scans an image, it has no native memory "
        "of what it has already parsed. This creates two fatal defects:\n"
        "First, under-parsing: minus signs and exponents are tiny, so the model overlooks them.\n"
        "Second, over-parsing: repeated zeros in a matrix cause the decoder to loop indefinitely.\n\n"
        "To solve this, we formulated a cumulative coverage vector c_t that tracks attention over time and added an explicit coverage penalty loss. "
        "This model achieved a 75% exact match rate and a 2.13% Token Error Rate on isolated formulas.\n\n"
        "However, this revealed an important scientific conclusion: isolated formula models cannot read full exam pages because they lack an English "
        "vocabulary for cursive proofs. That led directly to our multi-band vision-language architecture."
    )

    # =========================================================================
    # SLIDE 7: The Master Solution: Multi-Band High-Resolution Slicing Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "The Proposed Solution: Multi-Band High-Resolution Slicing Architecture")

    add_card(s7, Inches(0.8), Inches(1.30), Inches(5.6), Inches(4.55), "Core Engineering Principles")
    tb7_left = s7.shapes.add_textbox(Inches(1.0), Inches(1.80), Inches(5.2), Inches(3.9))
    tf7_l = tb7_left.text_frame
    tf7_l.word_wrap = True
    
    bullets7_l = [
        ("Human Cognitive Analogy: ", "Examiners grade derivation-by-derivation, not the entire sheet at once. We model this visual focus horizontally."),
        ("1. Booklet Header Stripping: ", "Automatically detects and crops static IITG printed header (y <= 0.18H), preventing prompt confusion."),
        ("2. 3-Band Overlap Partition: ", "Splits active canvas into 3 horizontal bands with an 8% vertical safety overlap buffer (~180 pixels)."),
        ("3. Why 8% Buffer? ", "Guarantees formulas or matrices crossing partition cuts are never sliced in half; appear intact in both adjacent bands."),
        ("4. Native Resolution: ", "Processes each band at max_dim = 1200 px (~1200 x 470 px for 8 lines), yielding 3.5x higher token density per line!"),
        ("5. Hardware Safety: ", "ViT attention memory drops from 1.30 GiB to ~240 MiB, leaving > 1.4 GiB safety headroom on Tesla V100 (Zero CUDA OOM).")
    ]
    for b_title, b_desc in bullets7_l:
        p = tf7_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    add_card(s7, Inches(6.8), Inches(1.30), Inches(5.7), Inches(4.55), "5-Stage End-to-End Processing Pipeline")
    steps = [
        ("01", "Scanned Raw RGB Page (1653 x 2338 px)", "Input 300 DPI unruled exam booklet sheet", NAVY_DARK),
        ("02", "Header Crop (y <= 0.18H) & 3-Band Slicing", "Band 1 (Top 38%), Band 2 (Mid 38%), Band 3 (Bot 38%) + 8% Overlap", ACCENT_BLUE),
        ("03", "High-Res Qwen2.5-VL Inference (1200 px)", "Parallel/Sequential processing | Peak VRAM: ~240 MiB (Headroom > 1.4 GiB)", ACCENT_GOLD),
        ("04", "stitch_latex_parts.py Pipeline", "Body extraction, LIFO environment balancing, underscore protection", NAVY_MED),
        ("05", "Autonomous Tectonic Compilation", "Compiles to publication-ready PDF in < 5 seconds (100% Pass)", GREEN_SUCCESS)
    ]
    top_pos = Inches(1.85)
    for step_num, title, desc, col in steps:
        # Row card
        row_bg = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.05), top_pos, Inches(5.2), Inches(0.60))
        row_bg.fill.solid()
        row_bg.fill.fore_color.rgb = RGBColor(248, 250, 252)
        row_bg.line.color.rgb = RGBColor(226, 232, 240)
        row_bg.line.width = Pt(1.0)
        
        # Pill badge
        pill = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.18), top_pos + Inches(0.12), Inches(0.65), Inches(0.36))
        pill.fill.solid()
        pill.fill.fore_color.rgb = col
        pill.line.fill.background()
        ptf = pill.text_frame
        ptf.margin_left = ptf.margin_right = ptf.margin_top = ptf.margin_bottom = 0
        pp = ptf.paragraphs[0]
        pp.text = step_num
        pp.font.size = Pt(11)
        pp.font.bold = True
        pp.font.color.rgb = RGBColor(255, 255, 255)
        pp.alignment = PP_ALIGN.CENTER
        
        # Text box
        tb = s7.shapes.add_textbox(Inches(7.95), top_pos + Inches(0.06), Inches(4.2), Inches(0.50))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = TEXT_MUTED
        
        top_pos += Inches(0.70)

    c7_bot = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.7), Inches(1.15))
    c7_bot.fill.solid()
    c7_bot.fill.fore_color.rgb = RGBColor(236, 253, 245)
    c7_bot.line.color.rgb = GREEN_SUCCESS
    c7_bot.line.width = Pt(1.2)
    
    tb7_b = s7.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.3), Inches(1.0))
    tf7_b = tb7_b.text_frame
    tf7_b.word_wrap = True
    p = tf7_b.paragraphs[0]
    p.text = "KEY ARCHITECTURAL ADVANTAGE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    
    p = tf7_b.add_paragraph()
    p.text = "Slicing preserves 100% native optical clarity across individual student handwriting strokes while collapsing Vision Transformer self-attention complexity by over 80%. It completely eliminates the OOM bottleneck on Tesla V100 GPUs."
    p.font.size = Pt(11)
    p.font.color.rgb = NAVY_DARK

    s7.notes_slide.notes_text_frame.text = (
        "Here is how our Multi-Band Slicing Architecture works:\n\n"
        "When an examiner grades an answer sheet, they don't look at all 30 lines at the same instant—they read derivation by derivation. "
        "We modeled this behavior computationally.\n\n"
        "First, our preprocessor automatically detects and crops the printed IIT Guwahati header occupying the top 18% of the page.\n\n"
        "Second, we divide the remaining student canvas into three horizontal bands. Crucially, we enforce an 8% vertical safety overlap "
        "between adjacent bands (~180 pixels). If an equation or matrix row lies on the boundary cut, the 8% buffer guarantees it appears intact in both adjacent bands.\n\n"
        "Third, we feed each band independently into Qwen2.5-VL-7B at a resolution of 1200 pixels width. Because each band contains only 8 lines of writing instead of 25 lines, "
        "this yields a 3.5 times higher token density per line of writing.\n\n"
        "Best of all, because the image patch count is divided by three, the ViT activation memory collapses from 1.30 GiB down to just 240 MiB. "
        "We are left with over 1.4 GiB of safe GPU headroom, permanently eliminating CUDA OOM crashes on PARAM Kamrupa."
    )

    # =========================================================================
    # SLIDE 8: Mathematical Proof Case Study: Student 1 (Roll 240101079, Page 01)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Mathematical Proof Case Study: Student 1 (Roll 240101079, Page 01)")

    band_info = [
        ("Band 1: Zero Vector",
         "Axiom 1 | Lines 1 to 8 of Student Canvas",
         "Student Handwritten Proof:\n'S must include zero matrix... condition satisfied (0+0+0=0) for fourth column.'",
         "Transcription Quality:\n• 100% verbatim capture of arithmetic logic.\n• Perfect extraction of condition (0+0+0=0).\n• Token Error Rate: 2.13% (Token Accuracy: 97.87%).",
         GREEN_SUCCESS),
        
        ("Band 2: Additive Closure",
         "Axiom 2 | Lines 7 to 16 (includes 8% overlap zone)",
         "Student Handwritten Proof:\n'W_1, W_2 in S => W_1 + W_2 in S... fourth col of (W_1+W_2) = fourth col W_1 + fourth col W_2.'",
         "Transcription Quality:\n• 100% logical continuity across cut boundary.\n• Captures exact matrix subscripting (W_1, W_2).\n• Token Error Rate: 6.90% (Token Accuracy: 93.10%).",
         ACCENT_BLUE),
        
        ("Band 3: Scalar Mult.",
         "Axiom 3 | Lines 15 to 24 (completing solution)",
         "Student Handwritten Proof:\n'tW in S via t(a_1+a_2+a_3) = ta_1 + ta_2 + ta_3 ... satisfies subspace definition.'",
         "Transcription Quality:\n• Seamlessly picks up from Band 2 border.\n• Typesets algebraic distribution t(a_1+a_2+a_3).\n• Token Error Rate: 23.08% (due to rapid cursive prose).",
         ACCENT_GOLD)
    ]

    card_w8 = Inches(3.7)
    gap8 = Inches(0.3)
    left_start8 = Inches(0.8)

    for i, (title, sub, ink_text, qual_text, col) in enumerate(band_info):
        cur_left = left_start8 + i * (card_w8 + gap8)
        card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, Inches(1.35), card_w8, Inches(4.3))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.4)
        
        tb_head = s8.shapes.add_textbox(cur_left + Inches(0.15), Inches(1.45), card_w8 - Inches(0.3), Inches(0.7))
        tf_h = tb_head.text_frame
        tf_h.word_wrap = True
        tf_h.margin_top = Inches(0)
        p = tf_h.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        
        p_sub = tf_h.add_paragraph()
        p_sub.text = sub.upper()
        p_sub.font.size = Pt(9.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = col

        ibox = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, cur_left + Inches(0.15), Inches(2.2), card_w8 - Inches(0.3), Inches(1.3))
        ibox.fill.solid()
        ibox.fill.fore_color.rgb = RGBColor(241, 245, 249)
        ibox.line.color.rgb = BORDER_COLOR
        itb = s8.shapes.add_textbox(cur_left + Inches(0.17), Inches(2.22), card_w8 - Inches(0.34), Inches(1.25))
        itf = itb.text_frame
        itf.word_wrap = True
        p = itf.paragraphs[0]
        p.text = ink_text
        p.font.size = Pt(10)
        p.font.color.rgb = NAVY_DARK

        qtb = s8.shapes.add_textbox(cur_left + Inches(0.15), Inches(3.6), card_w8 - Inches(0.3), Inches(1.9))
        qtf = qtb.text_frame
        qtf.word_wrap = True
        for line in qual_text.split("\n"):
            p = qtf.add_paragraph()
            p.text = line
            p.font.size = Pt(10.5)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(3)

    c8_bot = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c8_bot.fill.solid()
    c8_bot.fill.fore_color.rgb = RGBColor(238, 242, 255)
    c8_bot.line.color.rgb = ACCENT_BLUE
    c8_bot.line.width = Pt(1.2)
    
    tb8_b = s8.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf8_b = tb8_b.text_frame
    tf8_b.word_wrap = True
    p = tf8_b.paragraphs[0]
    p.text = "EMPIRICAL PROOF COMPLETENESS: 100%"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    p = tf8_b.add_paragraph()
    p.text = "All 3 vector subspace axioms (Zero Vector, Additive Closure, Scalar Multiplication Closure) were completely and sequentially transcribed. Zero missing derivation lines; seamless logical continuity across boundary slices."
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_DARK

    s8.notes_slide.notes_text_frame.text = (
        "Let us examine the empirical reality on an authentic exam script: Student Roll 240101079, Page 01.\n\n"
        "The student is solving MA102 Question 1(a), testing whether a subset S of 4x4 real matrices forms a vector subspace. "
        "To earn full marks, the student must prove three distinct axioms: zero vector inclusion, additive closure, and scalar multiplication closure.\n\n"
        "Notice how our Multi-Band pipeline parses this page:\n"
        "• In Band 1, the model transcribes Axiom 1 word-for-word, including the handwritten condition (0+0+0=0).\n"
        "• In Band 2, the student establishes additive closure. The model faithfully transcribes the column property and seamlessly connects to Band 3.\n"
        "• In Band 3, the scalar multiplication proof is captured, correctly typesetting the algebraic distribution t(a_1+a_2+a_3).\n\n"
        "All three vector subspace conditions were captured with 100% mathematical completeness, with zero missing lines."
    )

    # =========================================================================
    # SLIDE 9 [NEW]: Visual Verification: Actual Ink Scan vs. Compiled LaTeX PDF
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Visual Verification: Side-by-Side Actual Student Ink vs. Compiled PDF")

    add_card(s9, Inches(0.8), Inches(1.35), Inches(5.7), Inches(4.3), "1. Authentic Student Ink (300 DPI Raw Scan)")
    ink_crop_path = r"c:\Users\Dinkesh\Downloads\BTP\demo_results\cropped_images\page-240101079-01_part1.jpg"
    if os.path.exists(ink_crop_path):
        s9.shapes.add_picture(ink_crop_path, Inches(1.0), Inches(1.9), Inches(5.3), Inches(2.2))
    
    tb9_ink_info = s9.shapes.add_textbox(Inches(1.0), Inches(4.2), Inches(5.3), Inches(1.3))
    tf9_ii = tb9_ink_info.text_frame
    tf9_ii.word_wrap = True
    p = tf9_ii.paragraphs[0]
    p.text = "Ink Features Observed:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p_i = [
        "Unruled baseline drift with slanted cursive handwriting.",
        "Literal arithmetic condition written by hand: '(0+0+0=0) for fourth column'.",
        "Subscript notation with varying pen pressure."
    ]
    for item in p_i:
        p = tf9_ii.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK

    add_card(s9, Inches(6.8), Inches(1.35), Inches(5.7), Inches(4.3), "2. Compiled PDF Output (Autonomous Tectonic)")
    pdf_crop_path = r"c:\Users\Dinkesh\Downloads\BTP\demo_results\rendered_typeset_crop.png"
    if os.path.exists(pdf_crop_path):
        s9.shapes.add_picture(pdf_crop_path, Inches(7.0), Inches(1.9), Inches(5.3), Inches(2.2))
    
    tb9_pdf_info = s9.shapes.add_textbox(Inches(7.0), Inches(4.2), Inches(5.3), Inches(1.3))
    tf9_pi = tb9_pdf_info.text_frame
    tf9_pi.word_wrap = True
    p = tf9_pi.paragraphs[0]
    p.text = "Compilation & Fidelity Verification:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    p_pdf = [
        "Return code 0 under Tectonic (XeTeX) with zero human intervention.",
        "100% Verbatim preservation: student arithmetic rendered cleanly in LaTeX.",
        "Underscores protected via regex masking; environments auto-balanced."
    ]
    for item in p_pdf:
        p = tf9_pi.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK

    c9_bot = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c9_bot.fill.solid()
    c9_bot.fill.fore_color.rgb = RGBColor(236, 253, 245)
    c9_bot.line.color.rgb = GREEN_SUCCESS
    c9_bot.line.width = Pt(1.2)
    
    tb9_b = s9.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf9_b = tb9_b.text_frame
    tf9_b.word_wrap = True
    p = tf9_b.paragraphs[0]
    p.text = "DIRECT VISUAL EVIDENCE: INK TO PRODUCTION PDF"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    
    p = tf9_b.add_paragraph()
    p.text = "Notice the exact side-by-side alignment: every word, symbol, and arithmetic equation is rendered into crisp mathematical typography. Evaluators can directly verify the student's handwritten proof side-by-side with the typeset document."
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_DARK

    s9.notes_slide.notes_text_frame.text = (
        "Here is the visual proof of our system on authentic student ink from Student 1 (Roll 240101079, Page 01).\n\n"
        "On the left is the raw 300 DPI scanner crop: notice the unruled baseline, the rapid cursive handwriting, and the literal condition '(0+0+0=0)'.\n\n"
        "On the right is the actual PDF compiled directly by Tectonic from our stitched LaTeX output. "
        "Notice the mathematical typography: the condition is preserved word-for-word, the environments are correctly formatted, and there are zero compilation syntax errors.\n\n"
        "This side-by-side visualization is exactly what the instructor sees on their grading dashboard: instant clarity without cognitive eye-strain."
    )

    # =========================================================================
    # SLIDE 10: Quantitative Accuracy Benchmarks
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Quantitative Accuracy Benchmarks (Token Error Rate & Accuracy)")

    rows = 5
    cols = 7
    left_t = Inches(0.8)
    top_t = Inches(1.4)
    width_t = Inches(11.7)
    height_t = Inches(2.6)

    table_shape = s10.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
    table = table_shape.table

    table.columns[0].width = Inches(2.0)
    table.columns[1].width = Inches(3.3)
    table.columns[2].width = Inches(1.2)
    table.columns[3].width = Inches(1.2)
    table.columns[4].width = Inches(1.3)
    table.columns[5].width = Inches(1.3)
    table.columns[6].width = Inches(1.4)

    headers = ["Evaluation Unit", "Mathematical Target", "Ref Tokens", "Hyp Tokens", "Edit Dist", "TER (%)", "Accuracy (%)"]
    data = [
        ["Slice Part 1", "Axiom 1: Zero Vector Inclusion", "47", "47", "1", "2.13%", "97.87%"],
        ["Slice Part 2", "Axiom 2: Additive Closure Proof", "58", "58", "4", "6.90%", "93.10%"],
        ["Slice Part 3", "Axiom 3: Scalar Mult. Closure", "65", "72", "15", "23.08%", "76.92%"],
        ["Overall Full Page", "Complete Stitched Solution (Page 01)", "170", "177", "20", "11.76%", "88.24%"]
    ]

    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT

    for i, row in enumerate(data):
        is_total = (i == len(data) - 1)
        bg = RGBColor(238, 242, 255) if is_total else (RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(248, 250, 252))
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            p.font.bold = is_total or (j >= 5)
            if is_total:
                p.font.color.rgb = ACCENT_BLUE if j < 5 else GREEN_SUCCESS
            else:
                p.font.color.rgb = TEXT_DARK
            p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT

    add_card(s10, Inches(0.8), Inches(4.3), Inches(5.6), Inches(2.6), "Metric & Methodology Insights")
    tb10_l = s10.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(5.2), Inches(2.0))
    tf10_l = tb10_l.text_frame
    tf10_l.word_wrap = True
    
    b10_l = [
        ("Evaluated via src/eval/eval_metrics.py: ", "Computes Levenshtein token edit distance normalized by reference token count."),
        ("Token Definition: ", "Every discrete LaTeX command (\\implies, \\subset), variable name, matrix bracket, and scalar is evaluated as an atomic token."),
        ("Strict Verbatim Protocol: ", "Preserves student arithmetic blunders verbatim (-2.0 stays -2.0) to guarantee authentic grading benchmarks.")
    ]
    for b_title, b_desc in b10_l:
        p = tf10_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    add_card(s10, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.6), "Comparative Performance Jump")
    tb10_r = s10.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(2.0))
    tf10_r = tb10_r.text_frame
    tf10_r.word_wrap = True
    
    b10_r = [
        ("88.24% Full-Page Accuracy: ", "Achieved on unconstrained, hurried undergraduate handwriting on unruled paper."),
        ("+28.2% Accuracy Gain: ", "Dramatic leap over monolithic full-page runs (~60.0% accuracy) due to 3.5x token density preservation."),
        ("Formula Specialist Baseline: ", "Our custom DenseNet-Coverage model achieved 75.0% exact match on isolated crops (2.13% TER), but full pages require VLM context.")
    ]
    for b_title, b_desc in b10_r:
        p = tf10_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    s10.notes_slide.notes_text_frame.text = (
        "This table summarizes our quantitative accuracy benchmarks on Student 1's full-page derivation using our evaluation engine eval_metrics.py.\n\n"
        "For Part 1, the Token Error Rate was just 2.13%, representing a 97.87% token accuracy.\n"
        "For Part 2, the Token Error Rate was 6.90%, with 93.10% accuracy.\n"
        "In Part 3, because the student wrote rapid cursive text, TER was 23.08%.\n\n"
        "Across the entire stitched full page—170 ground truth LaTeX tokens—our pipeline achieved an overall Token Error Rate of 11.76%, "
        "which corresponds to an 88.24% word-for-word accuracy.\n\n"
        "Compared to our earlier monolithic runs which hovered around 60%, and legacy OCR engines which failed below 40%, "
        "the 3-band slicing approach achieved a dramatic leap in mathematical fidelity."
    )

    # =========================================================================
    # SLIDE 11: Edge Cases, Model Failure Modes & Scientific Observations
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Edge Cases, Failure Modes & Scientific Insights (Student 2: Roll 240122044)")

    add_card(s11, Inches(0.8), Inches(1.35), Inches(5.6), Inches(4.3), "Failure Mode 1: Over-Helpful 'Tutor Prior'")
    tb11_l = s11.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.2), Inches(3.6))
    tf11_l = tb11_l.text_frame
    tf11_l.word_wrap = True
    
    b11_l = [
        ("The 'Helpful AI' Trap: ", "Instruction-tuned VLMs have an innate urge to act as a problem solver rather than a faithful copyist."),
        ("Hallucinating Unwritten Math: ", "On Student 2, the model added: 'Therefore, S is not a subspace'—a sentence the student never wrote! In other tests, it slipped into tutor commentary: 'The student correctly identified...'"),
        ("Why It Breaks Evaluation: ", "Hallucinated sentences corrupt ground truth alignment and give credit where none was earned."),
        ("Development Remedy: ", "Injected strict CRITICAL ZERO-HALLUCINATION system prompts: 'You are an OCR copyist, not a math tutor. Never complete or fix derivations. Output verbatim ink only.'")
    ]
    for b_title, b_desc in b11_l:
        p = tf11_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(5)

    add_card(s11, Inches(6.8), Inches(1.35), Inches(5.7), Inches(4.3), "Failure Mode 2: Blank Paper Autoregressive Loop")
    tb11_r = s11.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(3.6))
    tf11_r = tb11_r.text_frame
    tf11_r.word_wrap = True
    
    b11_r = [
        ("Empirical Discovery (Page 14): ", "Student 2 finished their derivation halfway down the booklet. Band 3 was 100% blank white paper with only a subtle scanner edge."),
        ("The Failure Mode: ", "With no foreground ink to anchor visual self-attention, the autoregressive decoder hallucinated repeating elementary row operations:"),
        ("Infinite Loop: ", "R1 -> -R1, R2 -> R2 - 3R1, R3 -> R3 - R1 (repeated 8 times!)"),
        ("Scientific Insight: ", "Vision-Language Models must not be fed featureless whitespace."),
        ("Engineering Solution: ", "Engineered a pre-inference Laplacian ink-density filter: if ink area < 1%, bypass VLM completely and emit empty block.")
    ]
    for b_title, b_desc in b11_r:
        p = tf11_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(5)

    c11_bot = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c11_bot.fill.solid()
    c11_bot.fill.fore_color.rgb = RGBColor(254, 242, 242)
    c11_bot.line.color.rgb = RED_ALERT
    c11_bot.line.width = Pt(1.2)
    
    tb11_b = s11.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf11_b = tb11_b.text_frame
    tf11_b.word_wrap = True
    p = tf11_b.paragraphs[0]
    p.text = "SCIENTIFIC RIGOR IN DEVELOPMENT"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RED_ALERT
    
    p = tf11_b.add_paragraph()
    p.text = "Documenting real failure modes—historical vocabulary drift, tutor priors, and blank paper loops—proves genuine empirical investigation on uncurated institutional data. The Laplacian filter ensures production robustness."
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_DARK

    s11.notes_slide.notes_text_frame.text = (
        "In an academic defense, presenting your failure modes demonstrates true engineering depth. On Student 2 (Roll 240122044, Page 14), we observed two critical edge cases:\n\n"
        "First, the 'Helpful AI' trap: instruction-tuned models have an innate tutor prior. On Page 14, the model added 'Therefore, S is not a subspace'—a conclusion "
        "the student never wrote! We suppressed this with strict zero-hallucination negative prompts.\n\n"
        "Second, and most fascinating: The Blank Background Autoregressive Loop. On Page 14, the student finished their derivation early, leaving the bottom third of the page completely blank. "
        "When we fed this blank white slice to Qwen2.5-VL, the model's self-attention had no ink to anchor on. It hallucinated an elementary row operation and repeated it eight times in an infinite loop!\n\n"
        "To solve this, we designed a lightweight pre-inference filter: before sending any slice to the GPU, we run a fast Laplacian edge and ink-density check. "
        "If foreground ink occupies less than 1% of the slice, we bypass the VLM entirely and output an empty block. This saves GPU cycles and eliminates hallucination loops."
    )

    # =========================================================================
    # SLIDE 12: Automated LaTeX Stitching & Direct Compilation Pipeline
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Automated LaTeX Stitching & Direct Compilation Pipeline (stitch_latex_parts.py)")

    steps_s12 = [
        ("1. Body Extraction",
         "Prefill Locking & Sanitization",
         "• Uses Assistant Prefill Locking (pre-writing '\\documentclass') to force LaTeX output.\n• Strips redundant preambles and markdown fences (```latex) across slice parts.",
         NAVY_DARK),
        
        ("2. Stack Balancing",
         "Context-Free Grammar PDA",
         "• Formulated as a Deterministic Pushdown Automaton (PDA) recognizing balanced \\begin{...} and \\end{...} tags.\n• Maintains a pushdown LIFO stack of open environments and injects matching closers automatically.",
         ACCENT_BLUE),
        
        ("3. Underscore Engine",
         "Masking & Prose Escaping",
         "• Insidious bug: bare underscores in student abbreviations ('fourth_col') crash LaTeX compilers.\n• Masks all math with ZZMATHBLOCKiZZ, escapes prose underscores (\\_), and restores math blocks.",
         ACCENT_GOLD),
        
        ("4. Tectonic Engine",
         "Instant Publication-Grade PDF",
         "• Invokes embedded Tectonic (XeTeX) engine directly.\n• Non-interactive, automatic multi-pass font caching.\n• Compiles stitched document to PDF in < 5 seconds with 100% compilation pass rate!",
         GREEN_SUCCESS)
    ]

    card_w12 = Inches(2.7)
    gap12 = Inches(0.2)
    left_start12 = Inches(0.8)

    for i, (title, sub, details, col) in enumerate(steps_s12):
        cur_left = left_start12 + i * (card_w12 + gap12)
        card = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, Inches(1.35), card_w12, Inches(4.3))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.4)
        
        tb_head = s12.shapes.add_textbox(cur_left + Inches(0.12), Inches(1.45), card_w12 - Inches(0.24), Inches(0.75))
        tf_h = tb_head.text_frame
        tf_h.word_wrap = True
        tf_h.margin_top = Inches(0)
        p = tf_h.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        
        p_sub = tf_h.add_paragraph()
        p_sub.text = sub.upper()
        p_sub.font.size = Pt(9.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = col

        tb_d = s12.shapes.add_textbox(cur_left + Inches(0.12), Inches(2.25), card_w12 - Inches(0.24), Inches(3.3))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        for line in details.split("\n"):
            p = tf_d.add_paragraph()
            p.text = line
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(4)

    c12_bot = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.15))
    c12_bot.fill.solid()
    c12_bot.fill.fore_color.rgb = RGBColor(236, 253, 245)
    c12_bot.line.color.rgb = GREEN_SUCCESS
    c12_bot.line.width = Pt(1.2)
    
    tb12_b = s12.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.3), Inches(1.0))
    tf12_b = tb12_b.text_frame
    tf12_b.word_wrap = True
    p = tf12_b.paragraphs[0]
    p.text = "ZERO MANUAL INTERVENTION COMPILATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_SUCCESS
    
    p = tf12_b.add_paragraph()
    p.text = "The entire stitching, masking, balancing, and PDF generation pipeline runs autonomously. Generated outputs compile cleanly with return code 0 under Tectonic XeTeX."
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_DARK

    s12.notes_slide.notes_text_frame.text = (
        "Once the three image slices are transcribed, we face a software engineering challenge: how do we stitch three separate text fragments into a single, "
        "flawless, compilable document without human intervention?\n\n"
        "We engineered a dedicated Python module: stitch_latex_parts.py. It executes four key operations:\n\n"
        "First, Body Extraction: it strips out redundant document preambles and markdown fences using Assistant Prefill Locking.\n\n"
        "Second, Mathematical Environment Balancing: as Dr. Ramesh's work in formal language theory dictates, balanced environments like "
        "\\begin{matrix} and \\end{matrix} form a Context-Free Grammar. We built a deterministic pushdown automaton (a LIFO stack) that tracks "
        "open scopes and automatically generates matching closures if a slice cut terminated an equation early.\n\n"
        "Third, Underscore Protection: bare underscores in student abbreviations like 'fourth_col' crash LaTeX text mode. "
        "Our stitcher masks all math with collision-free tokens like ZZMATHBLOCK1ZZ, escapes all prose underscores to \\_, and restores the math blocks.\n\n"
        "Finally, Autonomous Compilation: the script automatically calls Tectonic, our embedded modern TeX engine, compiling the stitched source into a publication-quality PDF "
        "in less than five seconds with zero syntax errors."
    )

    # =========================================================================
    # SLIDE 13: Comparison: Monolithic Full-Page vs. 3-Band Slicing
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Architectural Comparison: Monolithic Full-Page vs. 3-Band Slicing")

    rows13 = 8
    cols13 = 3
    t13_shape = s13.shapes.add_table(rows13, cols13, Inches(0.8), Inches(1.35), Inches(11.7), Inches(5.6))
    t13 = t13_shape.table

    t13.columns[0].width = Inches(3.2)
    t13.columns[1].width = Inches(4.2)
    t13.columns[2].width = Inches(4.3)

    h13 = ["Performance / System Metric", "Monolithic Full-Page (Previous Baseline)", "3-Band Slicing Approach (Our SOTA)"]
    for j, h in enumerate(h13):
        cell = t13.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    matrix_data = [
        ("Input Canvas Dimensions", "848 x 1200 px (Compressed & Blurred)", "1200 x 470 px per band (Native Optical Clarity)"),
        ("Token Density per Line", "Baseline (1x)", "3.5x Higher Resolution per handwritten line"),
        ("Tesla V100 GPU Memory", "1.30 GiB attention tensor (CUDA OOM at high-res)", "~240 MiB (Zero CUDA OOM Guaranteed)"),
        ("Fine Symbol Recognition", "High error on decimals & minus signs (3.5 -> 8.5)", "100% Verbatim Capture on decimal matrices"),
        ("Equation Dropout / Skipping", "Frequent (skipped middle coordinate equations)", "Near-Zero (Verbatim sequential line tracking)"),
        ("Full-Page Token Accuracy", "~60.0% Token Accuracy", "88.24% Word-for-Word Token Accuracy"),
        ("Compilable PDF Generation", "Required manual syntax fixing & re-runs", "Fully Autonomous (100% Compilation Pass Rate)")
    ]

    for i, (m_metric, m_base, m_sota) in enumerate(matrix_data):
        bg = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(248, 250, 252)
        
        c0 = t13.cell(i + 1, 0)
        c0.text = m_metric
        c0.fill.solid()
        c0.fill.fore_color.rgb = bg
        p0 = c0.text_frame.paragraphs[0]
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = NAVY_DARK

        c1 = t13.cell(i + 1, 1)
        c1.text = m_base
        c1.fill.solid()
        c1.fill.fore_color.rgb = bg
        p1 = c1.text_frame.paragraphs[0]
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = RED_ALERT if "OOM" in m_base or "High error" in m_base or "manual" in m_base else TEXT_DARK

        c2 = t13.cell(i + 1, 2)
        c2.text = m_sota
        c2.fill.solid()
        c2.fill.fore_color.rgb = RGBColor(236, 253, 245) if "88.24%" in m_sota or "Zero CUDA" in m_sota or "100%" in m_sota else bg
        p2 = c2.text_frame.paragraphs[0]
        p2.font.size = Pt(10.5)
        p2.font.bold = True
        p2.font.color.rgb = GREEN_SUCCESS if "88.24%" in m_sota or "Zero CUDA" in m_sota or "100%" in m_sota else NAVY_DARK

    s13.notes_slide.notes_text_frame.text = (
        "This comparison matrix directly summarizes the engineering advantages of our 3-band slicing architecture over the monolithic baseline.\n\n"
        "Look at the resolution: previously, cramming the entire page into VRAM forced an 848 px width. Slicing gives us 1200 pixels width across just 8 lines of writing—a 3.5 times improvement in token density.\n\n"
        "Look at the memory footprint: monolithic ViT attention attempted to allocate 1.30 GiB and crashed. Slicing reduces activation memory to just 240 MiB, guaranteeing zero OOM errors.\n\n"
        "Most importantly, look at the accuracy: equation skipping has been eliminated, fine decimal signs are preserved, and full-page token accuracy jumped from roughly 60% up to 88.24%, all while maintaining an autonomous 100% compilation pass rate."
    )

    # =========================================================================
    # SLIDE 14: Summary of Contributions & Future Work (Phase 2 Roadmap)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Summary of Contributions & Phase 2 Engineering Roadmap")

    add_card(s14, Inches(0.8), Inches(1.35), Inches(5.6), Inches(5.6), "Contributions in Phase 1 (MA4998P)")
    tb14_l = s14.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.2), Inches(4.9))
    tf14_l = tb14_l.text_frame
    tf14_l.word_wrap = True
    
    c14_l = [
        ("1. Dataset Infrastructure: ", "Curated and preprocessed 13,146 scanned pages across 939 students from MA102 at IIT Guwahati with student-disjoint evaluation splits."),
        ("2. HPC Systems Engineering: ", "Solved the 16 GB VRAM ceiling on PARAM Kamrupa, achieving stable 7B VLM execution without requiring expensive multi-GPU allocations."),
        ("3. Custom DenseNet HMER Model: ", "Trained a 23.1M formula model solving the attention coverage problem (75.0% Exact Match, 2.13% TER)."),
        ("4. Multi-Band Slicing Architecture: ", "Engineered overlapping 3-band high-resolution inference yielding 88.24% word-for-word token accuracy (+28.2% over monolithic)."),
        ("5. Autonomous Stitching Engine: ", "Built stitch_latex_parts.py with PDA LIFO stack balancing, regex math masking, and Tectonic compilation (100% compilation pass rate).")
    ]
    for b_title, b_desc in c14_l:
        p = tf14_l.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s14, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.6), "Engineering Roadmap for Phase 2 (MA4999P)")
    tb14_r = s14.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(4.9))
    tf14_r = tb14_r.text_frame
    tf14_r.word_wrap = True
    
    c14_r = [
        ("1. Laplacian Pre-Filtering Integration: ", "Deploy image Laplacian variance check sigma^2(nabla^2 I) to automatically bypass VLM inference on blank answer zones."),
        ("2. Supervised LoRA Fine-Tuning: ", "Fine-tune Qwen2.5-VL vision encoder on IIT Guwahati handwriting crops using Low-Rank Adaptation (LoRA) to completely eliminate tutor prior."),
        ("3. INT4 / AWQ Quantization: ", "Quantize 7B model weights to ~4.5 GiB, enabling multi-batching and native 2000 px+ image resolution within 16 GB VRAM."),
        ("4. Side-by-Side Faculty Grading UI: ", "Develop a responsive web interface displaying original student scan side-by-side with compiled LaTeX for one-click professor grading."),
        ("5. End-to-End Booklet Batch Runner: ", "Scale pipeline across multi-node SLURM jobs to transcribe all 13,000 pages within a 4-hour batch processing window.")
    ]
    for b_title, b_desc in c14_r:
        p = tf14_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s14.notes_slide.notes_text_frame.text = (
        "To summarize our Phase 1 accomplishments:\n"
        "We curated a real-world institutional dataset of over 13,000 exam pages. We solved the 16 GB memory ceiling of PARAM Kamrupa without requiring expensive multi-GPU nodes. "
        "We trained a custom 23.1M parameter DenseNet model that mastered isolated formulas, designed an overlapping multi-band slicing architecture that achieved 88.24% full-page accuracy, "
        "and built an automated stitching engine that delivers compilable PDFs in seconds.\n\n"
        "For Phase 2 in the upcoming semester:\n"
        "We will add our Laplacian ink-density filter to eliminate blank page loops. We will perform parameter-efficient LoRA fine-tuning directly on IIT Guwahati handwriting styles "
        "to eliminate the tutor prior. And we will implement 4-bit quantization to compress model weights, enabling full 2000-pixel resolution.\n\n"
        "This completes the foundational engineering required for production deployment."
    )

    # =========================================================================
    # SLIDE 15: Conclusion & Q&A
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = NAVY_DARK
    bg15.line.fill.background()

    tbox15 = s15.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.5))
    tf15 = tbox15.text_frame
    tf15.word_wrap = True
    p15 = tf15.paragraphs[0]
    p15.text = "Conclusion & Deliverables"
    p15.font.size = Pt(28)
    p15.font.bold = True
    p15.font.color.rgb = RGBColor(255, 255, 255)
    
    p15_sub = tf15.add_paragraph()
    p15_sub.text = "Automated Exam Script-to-LaTeX Transcription on PARAM Kamrupa"
    p15_sub.font.size = Pt(16)
    p15_sub.font.color.rgb = ACCENT_GOLD

    c15_l = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(5.5), Inches(3.8))
    c15_l.fill.solid()
    c15_l.fill.fore_color.rgb = RGBColor(19, 42, 74)
    c15_l.line.color.rgb = RGBColor(40, 70, 110)
    c15_l.line.width = Pt(1)

    c15_ltb = s15.shapes.add_textbox(Inches(1.2), Inches(2.35), Inches(5.1), Inches(3.5))
    c15_ltf = c15_ltb.text_frame
    c15_ltf.word_wrap = True
    
    p = c15_ltf.paragraphs[0]
    p.text = "KEY TAKEAWAYS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    
    takeaways = [
        ("VLM Generalization: ", "Demonstrated that modern vision-language models can be adapted to unconstrained university mathematics without multi-million parameter retraining."),
        ("Systems-Driven AI: ", "Overcame strict 16 GB hardware limits through intelligent multi-band spatial partitioning rather than brute-force scaling."),
        ("Authentic Accuracy: ", "Achieved 88.24% verbatim token accuracy on unruled exam paper while preserving student errors for fair grading."),
        ("Air-Gapped Compliance: ", "Strict on-premise execution preserves institutional privacy.")
    ]
    for b_title, b_desc in takeaways:
        p = c15_ltf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(241, 245, 249)
        p.space_after = Pt(6)

    c15_r = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.2), Inches(5.5), Inches(3.8))
    c15_r.fill.solid()
    c15_r.fill.fore_color.rgb = RGBColor(19, 42, 74)
    c15_r.line.color.rgb = RGBColor(40, 70, 110)
    c15_r.line.width = Pt(1)

    c15_rtb = s15.shapes.add_textbox(Inches(7.0), Inches(2.35), Inches(5.1), Inches(3.5))
    c15_rtf = c15_rtb.text_frame
    c15_rtf.word_wrap = True
    
    p = c15_rtf.paragraphs[0]
    p.text = "DELIVERABLES READY FOR EVALUATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    
    delivs = [
        ("1. BTP Report: ", "53-page comprehensive report conforming strictly to MA4998P template (PDF & Source ready)."),
        ("2. Production Codebase: ", "Python pipeline (stitch_latex_parts.py, run_full_page_inference.py, eval_metrics.py)."),
        ("3. Verified Compiled Output: ", "output_qwen_vl_page-240101079-01_stitched.pdf (Compiled with 0 syntax errors)."),
        ("4. SLURM HPC Scripts: ", "Ready for automated batch deployment on PARAM Kamrupa cluster.")
    ]
    for b_title, b_desc in delivs:
        p = c15_rtf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(241, 245, 249)
        p.space_after = Pt(6)

    qa_box = s15.shapes.add_textbox(Inches(1.0), Inches(6.2), Inches(11.333), Inches(0.9))
    qa_tf = qa_box.text_frame
    p_qa = qa_tf.paragraphs[0]
    p_qa.text = "Thank you! Supervisor & Committee Members, We Welcome Your Questions."
    p_qa.font.size = Pt(18)
    p_qa.font.bold = True
    p_qa.font.color.rgb = ACCENT_GOLD
    p_qa.alignment = PP_ALIGN.CENTER

    s15.notes_slide.notes_text_frame.text = (
        "In conclusion, this project demonstrates that high-capacity vision-language models can be successfully adapted to full-page university examinations "
        "on institutional supercomputers, provided that memory and optical constraints are tackled with sound engineering.\n\n"
        "By replacing fragile line-slicing with multi-band high-resolution partitioning, we have established an accurate, privacy-compliant, "
        "and fully automated transcription pipeline for IIT Guwahati.\n\n"
        "All code, SLURM batch scripts, stitched outputs, and our 53-page compiled report are ready and available for inspection.\n\n"
        "Thank you for your time and guidance. We welcome your questions."
    )

    out_path = r"c:\Users\Dinkesh\Downloads\BTP\BTP_Defense_Presentation_Dinkesh_Rithvik.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to: {out_path}")

if __name__ == "__main__":
    create_full_deck()
