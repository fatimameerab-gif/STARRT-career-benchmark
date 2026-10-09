"""Career label set for the STARRT.ai paper, and mappings from each data source.

13 classes in 4 parent groups. Each class is anchored to O*NET-SOC codes and ESCO
occupation labels so the taxonomy is defensible rather than invented.
"""

# class -> (parent group, O*NET-SOC codes, ESCO preferred labels)
CLASSES = {
    "Web Development":            ("Software Development", ["15-1254.00"], ["web developer"]),
    "Mobile Development":         ("Software Development", ["15-1252.00"], ["mobile application developer"]),
    "Game Development":           ("Software Development", ["15-1255.01"], ["digital games developer"]),
    "Embedded & IoT":             ("Software Development", ["17-2061.00"], ["embedded systems software developer", "embedded system designer"]),
    "Data Science & Analytics":   ("Data & AI", ["15-2051.00", "15-2051.01"], ["data scientist"]),
    "Data Engineering & Databases": ("Data & AI", ["15-1243.00", "15-1243.01", "15-1242.00"], ["data engineer", "database developer", "database administrator"]),
    "AI/ML Engineering":          ("Data & AI", [], ["artificial intelligence engineer"]),
    "DevOps & Cloud":             ("Infrastructure & Security", ["15-1299.08"], ["cloud DevOps engineer", "cloud engineer"]),
    "Cybersecurity":              ("Infrastructure & Security", ["15-1212.00", "15-1299.04", "15-1299.05"], ["ICT security administrator"]),
    "Networking & SysAdmin":      ("Infrastructure & Security", ["15-1244.00", "15-1241.00"], ["ICT network administrator"]),
    "QA & Testing":               ("Quality, Design & Management", ["15-1253.00"], ["software tester"]),
    "UI/UX Design":               ("Quality, Design & Management", ["15-1255.00"], ["user interface designer", "user experience analyst"]),
    "Product & Project Management": ("Quality, Design & Management", ["15-1299.09", "13-1082.00"], ["ICT project manager", "ICT product manager"]),
}

# Stack Overflow 2025 DevType -> class (None = excluded: student, generic, non-technical, ambiguous)
SO_DEVTYPE = {
    "Developer, full-stack": "Web Development",
    "Developer, back-end": "Web Development",
    "Developer, front-end": "Web Development",
    "Developer, mobile": "Mobile Development",
    "Developer, game or graphics": "Game Development",
    "Developer, embedded applications or devices": "Embedded & IoT",
    "Data scientist": "Data Science & Analytics",
    "Data or business analyst": "Data Science & Analytics",
    "Data engineer": "Data Engineering & Databases",
    "Database administrator or engineer": "Data Engineering & Databases",
    "AI/ML engineer": "AI/ML Engineering",
    "Developer, AI apps or physical AI": "AI/ML Engineering",
    "DevOps engineer or professional": "DevOps & Cloud",
    "Cloud infrastructure engineer": "DevOps & Cloud",
    "Cybersecurity or InfoSec professional": "Cybersecurity",
    "System administrator": "Networking & SysAdmin",
    "Developer, QA or test": "QA & Testing",
    "UX, Research Ops or UI design professional": "UI/UX Design",
    "Product manager": "Product & Project Management",
    "Project manager": "Product & Project Management",
}

# 2023 and 2024 surveys used slightly different DevType names.
# "Data scientist or machine learning specialist" mixes two of our classes, so it is excluded.
SO_DEVTYPE_2023_2024 = {
    "Developer, full-stack": "Web Development",
    "Developer, back-end": "Web Development",
    "Developer, front-end": "Web Development",
    "Developer, mobile": "Mobile Development",
    "Developer, game or graphics": "Game Development",
    "Developer, embedded applications or devices": "Embedded & IoT",
    "Data or business analyst": "Data Science & Analytics",
    "Engineer, data": "Data Engineering & Databases",
    "Data engineer": "Data Engineering & Databases",
    "Database administrator": "Data Engineering & Databases",
    "Developer, AI": "AI/ML Engineering",
    "DevOps specialist": "DevOps & Cloud",
    "Cloud infrastructure engineer": "DevOps & Cloud",
    "Engineer, site reliability": "DevOps & Cloud",
    "Security professional": "Cybersecurity",
    "System administrator": "Networking & SysAdmin",
    "Developer, QA or test": "QA & Testing",
    "Designer": "UI/UX Design",
    "Product manager": "Product & Project Management",
    "Project manager": "Product & Project Management",
}

# Faruque et al. (2024) careers -> class. "Development" and "Software Development and
# Engineering" are too broad for one class, so they stay unmapped (kept only for the
# separate 6-class benchmark that replicates Faruque et al.).
FARUQUE = {
    "Artificial Intelligence": "AI/ML Engineering",
    "Data Science": "Data Science & Analytics",
    "Security": "Cybersecurity",
    "User Experience (UX) and User Interface (UI) Design": "UI/UX Design",
    "Development": None,
    "Software Development and Engineering": None,
}

# Old STARRT.ai Google Form field -> class
FORMS = {
    "AI / Machine Learning Engineering": "AI/ML Engineering",
    "Web Development (Frontend / Backend / Fullstack)": "Web Development",
    "Mobile Development (iOS / Android)": "Mobile Development",
    "Data Science & Analytics": "Data Science & Analytics",
    "UI/UX Design / Product Design": "UI/UX Design",
    "Product Management (tech)": "Product & Project Management",
    "Quality Assurance / Testing": "QA & Testing",
    "Cybersecurity / Information Security": "Cybersecurity",
    "Cloud Engineering (AWS/GCP/Azure)": "DevOps & Cloud",
    "DevOps / Site Reliability Engineering": "DevOps & Cloud",
    "Network management": "Networking & SysAdmin",
    "Computer Networks": "Networking & SysAdmin",
    "Embedded Systems / IoT": "Embedded & IoT",
    "Academia / Research (Computer Science)": None,
    "Software Engineering": None,
    "Operating Systems": None,
}

# Djinni (Drushchak & Romanyshyn 2024) Primary Keyword -> class.
# Programming-language keywords that span several roles (Java, .NET, Python, C++, Golang,
# Scala, Rust) and non-engineering keywords are excluded (None). AI/ML has no keyword, so
# Data Science / Python profiles whose job title (Position) names ML or AI are relabelled
# AI/ML Engineering in build_datasets.py; Position itself is never used as model input.
DJINNI = {
    "JavaScript": "Web Development", "PHP": "Web Development", "Node.js": "Web Development", "Ruby": "Web Development",
    "iOS": "Mobile Development", "Android": "Mobile Development", "Flutter": "Mobile Development",
    "Unity": "Game Development",
    "Data Science": "Data Science & Analytics", "Data Analyst": "Data Science & Analytics",
    "Data Engineer": "Data Engineering & Databases", "SQL": "Data Engineering & Databases",
    "DevOps": "DevOps & Cloud",
    "Security": "Cybersecurity",
    "Sysadmin": "Networking & SysAdmin",
    "QA": "QA & Testing", "QA Automation": "QA & Testing",
    "Design": "UI/UX Design",
    "Project Manager": "Product & Project Management", "Product Manager": "Product & Project Management",
    "Scrum Master": "Product & Project Management",
}
DJINNI_ML_TITLE = r"(?i)\b(machine learning|deep learning|ml|ai|computer vision|cv engineer|nlp|llm|mlops)\b"
