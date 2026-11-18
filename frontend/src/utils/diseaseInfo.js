/**
 * diseaseInfo.js
 * Rich metadata for all 38 PlantVillage disease classes.
 * Each entry contains display info, severity, and actionable treatment advice.
 */

export const DISEASE_DB = {
  // ── Apple ──────────────────────────────────────────────────────────────────
  "Apple___Apple_scab": {
    displayName: "Apple Scab",
    plant: "Apple",
    severity: "medium",
    description: "A fungal disease causing olive-green to brown velvety spots on leaves and fruit, eventually leading to defoliation and reduced yield.",
    treatment: ["Apply fungicides (captan or myclobutanil) at 7–10 day intervals during wet spring weather.", "Remove and destroy fallen infected leaves to break the disease cycle.", "Plant scab-resistant apple varieties when possible."],
    emoji: "🍎",
  },
  "Apple___Black_rot": {
    displayName: "Black Rot",
    plant: "Apple",
    severity: "high",
    description: "A serious fungal disease (Botryosphaeria obtusa) causing leaf spots, fruit rot, and cankers on branches. Spreads rapidly in warm, humid conditions.",
    treatment: ["Prune and destroy all infected wood and mummified fruit.", "Apply copper-based or captan fungicides from bloom to harvest.", "Maintain good air circulation through proper tree spacing and pruning."],
    emoji: "🍎",
  },
  "Apple___Cedar_apple_rust": {
    displayName: "Cedar Apple Rust",
    plant: "Apple",
    severity: "medium",
    description: "A fungal disease requiring both apple and cedar/juniper to complete its life cycle. Causes bright orange-yellow spots on leaves.",
    treatment: ["Remove nearby cedar or juniper trees where feasible.", "Apply myclobutanil or mancozeb sprays starting at pink bud stage.", "Plant rust-resistant apple cultivars."],
    emoji: "🍎",
  },
  "Apple___healthy": {
    displayName: "Healthy",
    plant: "Apple",
    severity: "none",
    description: "This apple leaf shows no signs of disease or pest damage. The plant appears to be in excellent health.",
    treatment: ["Continue regular watering and balanced fertilization.", "Monitor for early signs of pests or disease.", "Ensure good air circulation around the tree."],
    emoji: "🍎",
  },

  // ── Blueberry ──────────────────────────────────────────────────────────────
  "Blueberry___healthy": {
    displayName: "Healthy",
    plant: "Blueberry",
    severity: "none",
    description: "Your blueberry plant looks healthy with no visible signs of disease or nutrient deficiency.",
    treatment: ["Maintain acidic soil pH between 4.5 and 5.5.", "Mulch with pine bark or wood chips to retain moisture."],
    emoji: "🫐",
  },

  // ── Cherry ─────────────────────────────────────────────────────────────────
  "Cherry_(including_sour)___Powdery_mildew": {
    displayName: "Powdery Mildew",
    plant: "Cherry",
    severity: "medium",
    description: "A fungal disease appearing as white powdery spots on leaves, shoots, and sometimes fruit. Thrives in warm, dry conditions with high humidity.",
    treatment: ["Apply sulfur-based or potassium bicarbonate fungicides at first sign of infection.", "Improve air circulation with strategic pruning.", "Avoid overhead irrigation; water at the base."],
    emoji: "🍒",
  },
  "Cherry_(including_sour)___healthy": {
    displayName: "Healthy",
    plant: "Cherry",
    severity: "none",
    description: "This cherry leaf is healthy with vibrant color and no signs of disease.",
    treatment: ["Continue standard care practices.", "Watch for cherry leaf spot in wet spring seasons."],
    emoji: "🍒",
  },

  // ── Corn (Maize) ───────────────────────────────────────────────────────────
  "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
    displayName: "Gray Leaf Spot",
    plant: "Corn",
    severity: "high",
    description: "One of the most significant corn diseases worldwide. Caused by Cercospora zeae-maydis, producing gray to tan rectangular lesions parallel to leaf veins.",
    treatment: ["Plant resistant hybrids — this is the most effective control method.", "Apply foliar fungicides (strobilurins or triazoles) at VT/R1 growth stage.", "Rotate crops and till residue to reduce inoculum."],
    emoji: "🌽",
  },
  "Corn_(maize)___Common_rust_": {
    displayName: "Common Rust",
    plant: "Corn",
    severity: "medium",
    description: "Caused by Puccinia sorghi, producing small, oval to elongated brick-red pustules on both leaf surfaces. Severity increases in cool, moist weather.",
    treatment: ["Scout fields frequently from V6 onwards.", "Apply triazole or strobilurin fungicides if disease pressure is high before tasseling.", "Use resistant hybrid varieties in future plantings."],
    emoji: "🌽",
  },
  "Corn_(maize)___Northern_Leaf_Blight": {
    displayName: "Northern Leaf Blight",
    plant: "Corn",
    severity: "high",
    description: "Caused by Exserohilum turcicum. Long, cigar-shaped gray-green to tan lesions appear on lower leaves first and progress upward. Can cause 30–50% yield loss in severe cases.",
    treatment: ["Apply foliar fungicides (propiconazole, azoxystrobin) at disease onset.", "Plant resistant hybrids with Ht1, Ht2, or Ht3 resistance genes.", "Reduce residue through tillage and crop rotation."],
    emoji: "🌽",
  },
  "Corn_(maize)___healthy": {
    displayName: "Healthy",
    plant: "Corn",
    severity: "none",
    description: "This corn plant appears healthy with no signs of disease or stress.",
    treatment: ["Ensure adequate nitrogen nutrition, especially at V6 stage.", "Monitor for early disease and pest pressure regularly."],
    emoji: "🌽",
  },

  // ── Grape ──────────────────────────────────────────────────────────────────
  "Grape___Black_rot": {
    displayName: "Black Rot",
    plant: "Grape",
    severity: "high",
    description: "A destructive fungal disease (Guignardia bidwellii) causing brown circular leaf spots with dark borders and rapidly rotting fruit that shrivels into black mummies.",
    treatment: ["Apply mancozeb or myclobutanil fungicides starting at bud swell and continue every 7–14 days.", "Remove mummified fruit and infected plant material.", "Ensure good canopy management for air circulation."],
    emoji: "🍇",
  },
  "Grape___Esca_(Black_Measles)": {
    displayName: "Esca (Black Measles)",
    plant: "Grape",
    severity: "high",
    description: "A complex trunk disease caused by multiple wood-decay fungi. Symptoms include tiger-stripe leaf discoloration, stunted shoots, and sudden vine death in severe cases.",
    treatment: ["There is no cure; manage by removing and burning infected wood.", "Protect pruning wounds with fungicide paste (thiophanate-methyl).", "Avoid water stress which triggers symptom expression."],
    emoji: "🍇",
  },
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
    displayName: "Leaf Blight",
    plant: "Grape",
    severity: "medium",
    description: "Caused by Pseudocercospora vitis, producing irregular dark brown lesions on older leaves. Can cause premature defoliation in severe cases.",
    treatment: ["Apply copper-based fungicides or mancozeb preventatively.", "Improve air flow through canopy training and pruning.", "Avoid excessive nitrogen fertilization."],
    emoji: "🍇",
  },
  "Grape___healthy": {
    displayName: "Healthy",
    plant: "Grape",
    severity: "none",
    description: "Your grapevine leaves show no signs of disease or nutrient deficiency.",
    treatment: ["Maintain consistent canopy management throughout the season.", "Scout regularly for powdery mildew and downy mildew, which are very common."],
    emoji: "🍇",
  },

  // ── Orange ─────────────────────────────────────────────────────────────────
  "Orange___Haunglongbing_(Citrus_greening)": {
    displayName: "Citrus Greening (HLB)",
    plant: "Orange",
    severity: "critical",
    description: "The most devastating citrus disease worldwide, caused by the bacterium Candidatus Liberibacter and spread by the Asian citrus psyllid. There is currently no cure. Causes blotchy yellowing, misshapen bitter fruit, and eventual tree death.",
    treatment: ["⚠️ Report immediately to your local agricultural authority — this is a regulated disease.", "Remove and destroy infected trees to prevent spread.", "Control the Asian citrus psyllid vector with insecticides.", "There is no treatment; focus on prevention and containment."],
    emoji: "🍊",
  },

  // ── Peach ──────────────────────────────────────────────────────────────────
  "Peach___Bacterial_spot": {
    displayName: "Bacterial Spot",
    plant: "Peach",
    severity: "medium",
    description: "Caused by Xanthomonas arboricola pv. pruni, causing water-soaked spots that turn purple/brown on leaves and fruit, often with shot-hole appearance.",
    treatment: ["Apply copper hydroxide bactericides during dormancy and the growing season.", "Plant resistant varieties; avoid highly susceptible cultivars.", "Avoid injury to trees which provides entry points for the bacterium."],
    emoji: "🍑",
  },
  "Peach___healthy": {
    displayName: "Healthy",
    plant: "Peach",
    severity: "none",
    description: "This peach leaf is healthy with no signs of bacterial or fungal disease.",
    treatment: ["Prune for open canopy structure to maximize air circulation.", "Apply dormant copper sprays preventatively each season."],
    emoji: "🍑",
  },

  // ── Bell Pepper ────────────────────────────────────────────────────────────
  "Pepper,_bell___Bacterial_spot": {
    displayName: "Bacterial Spot",
    plant: "Bell Pepper",
    severity: "medium",
    description: "Caused by Xanthomonas euvesicatoria, producing water-soaked lesions that turn brown with yellow halos. Spreads rapidly in warm, rainy conditions.",
    treatment: ["Use certified disease-free seed and transplants.", "Apply copper bactericides preventatively; avoid when temperatures exceed 35°C.", "Practice a 2–3 year crop rotation."],
    emoji: "🫑",
  },
  "Pepper,_bell___healthy": {
    displayName: "Healthy",
    plant: "Bell Pepper",
    severity: "none",
    description: "Your bell pepper plant looks healthy and disease-free.",
    treatment: ["Ensure consistent moisture to prevent blossom end rot.", "Scout regularly for early signs of bacterial spot and aphids."],
    emoji: "🫑",
  },

  // ── Potato ─────────────────────────────────────────────────────────────────
  "Potato___Early_blight": {
    displayName: "Early Blight",
    plant: "Potato",
    severity: "medium",
    description: "Caused by Alternaria solani. Dark brown to black concentric ring lesions (bullseye pattern) appear on older leaves first, progressing upward. Weakens plants and reduces tuber size.",
    treatment: ["Apply chlorothalonil, mancozeb, or azoxystrobin fungicides preventatively.", "Ensure adequate potassium nutrition to improve plant resistance.", "Remove and destroy heavily infected foliage; rotate crops annually."],
    emoji: "🥔",
  },
  "Potato___Late_blight": {
    displayName: "Late Blight",
    plant: "Potato",
    severity: "critical",
    description: "Caused by Phytophthora infestans — the same pathogen responsible for the Irish Potato Famine. Water-soaked dark lesions spread explosively in cool, moist weather, destroying entire fields.",
    treatment: ["⚠️ Act immediately — this spreads very fast.", "Apply systemic fungicides (metalaxyl + mancozeb) at first symptom.", "Destroy infected plant material; do not compost.", "Use certified blight-resistant potato varieties."],
    emoji: "🥔",
  },
  "Potato___healthy": {
    displayName: "Healthy",
    plant: "Potato",
    severity: "none",
    description: "This potato plant appears healthy with no signs of blight or disease.",
    treatment: ["Hill potatoes to prevent tuber greening.", "Monitor moisture levels; over-irrigation encourages late blight."],
    emoji: "🥔",
  },

  // ── Raspberry ──────────────────────────────────────────────────────────────
  "Raspberry___healthy": {
    displayName: "Healthy",
    plant: "Raspberry",
    severity: "none",
    description: "Your raspberry plant is healthy with no visible signs of disease.",
    treatment: ["Prune out old floricanes after fruiting to improve air circulation.", "Watch for cane diseases (spur blight, cane blight) in subsequent seasons."],
    emoji: "🫐",
  },

  // ── Soybean ────────────────────────────────────────────────────────────────
  "Soybean___healthy": {
    displayName: "Healthy",
    plant: "Soybean",
    severity: "none",
    description: "This soybean plant looks healthy with no signs of disease or stress.",
    treatment: ["Scout regularly for soybean rust and sudden death syndrome.", "Ensure adequate potassium and phosphorus at V3 stage."],
    emoji: "🌱",
  },

  // ── Squash ─────────────────────────────────────────────────────────────────
  "Squash___Powdery_mildew": {
    displayName: "Powdery Mildew",
    plant: "Squash",
    severity: "medium",
    description: "A very common fungal disease causing white powdery coating on leaf surfaces. Unlike most fungi, it thrives in warm, dry conditions and can quickly cover entire plants.",
    treatment: ["Apply potassium bicarbonate, neem oil, or sulfur fungicides at first sign.", "Space plants for good air circulation and avoid overhead watering.", "Remove severely infected leaves to slow the spread."],
    emoji: "🥒",
  },

  // ── Strawberry ─────────────────────────────────────────────────────────────
  "Strawberry___Leaf_scorch": {
    displayName: "Leaf Scorch",
    plant: "Strawberry",
    severity: "medium",
    description: "Caused by Diplocarpon earliana, producing irregular purple spots that coalesce, giving leaves a scorched appearance and causing early defoliation.",
    treatment: ["Apply captan or myclobutanil fungicides from early spring through harvest.", "Remove old infected foliage after renovation.", "Plant in well-drained soil and avoid over-crowding."],
    emoji: "🍓",
  },
  "Strawberry___healthy": {
    displayName: "Healthy",
    plant: "Strawberry",
    severity: "none",
    description: "Your strawberry plant is healthy with bright green leaves and no signs of disease.",
    treatment: ["Renovate beds immediately after harvest to reduce disease carryover.", "Monitor for two-spotted spider mites in hot, dry weather."],
    emoji: "🍓",
  },

  // ── Tomato ─────────────────────────────────────────────────────────────────
  "Tomato___Bacterial_spot": {
    displayName: "Bacterial Spot",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by multiple Xanthomonas species. Small, water-soaked lesions with yellow halos appear on leaves and fruit, causing defoliation and unmarketable fruit.",
    treatment: ["Apply copper hydroxide + mancozeb tank-mix from transplanting.", "Use disease-free certified seed; avoid working with wet plants.", "Rotate out of tomatoes for 1–2 years."],
    emoji: "🍅",
  },
  "Tomato___Early_blight": {
    displayName: "Early Blight",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by Alternaria solani. Concentric dark rings form classic 'bullseye' lesions on older leaves. Favored by warm temperatures and alternating wet and dry periods.",
    treatment: ["Apply chlorothalonil or azoxystrobin fungicides preventatively every 7–10 days.", "Mulch around plants to reduce soil splash.", "Remove lower infected leaves and improve plant spacing."],
    emoji: "🍅",
  },
  "Tomato___Late_blight": {
    displayName: "Late Blight",
    plant: "Tomato",
    severity: "critical",
    description: "Caused by Phytophthora infestans. Irregular, greasy-looking dark lesions spread rapidly in cool, moist weather. White sporulation may be visible on the underside of leaves.",
    treatment: ["⚠️ Respond immediately — this disease can destroy an entire crop in days.", "Apply metalaxyl-based systemic fungicides at first sign.", "Remove and bag all infected plant material; do not compost.", "Consider removing highly infected plants entirely to protect remaining plants."],
    emoji: "🍅",
  },
  "Tomato___Leaf_Mold": {
    displayName: "Leaf Mold",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by Passalora fulva (formerly Fulvia fulva). Yellow patches form on the upper leaf surface with olive-green to brown velvety growth on the underside. A common greenhouse tomato disease.",
    treatment: ["Reduce relative humidity below 85% through ventilation.", "Apply chlorothalonil or copper-based fungicides preventatively.", "Remove and destroy infected leaves; avoid working in plants when wet."],
    emoji: "🍅",
  },
  "Tomato___Septoria_leaf_spot": {
    displayName: "Septoria Leaf Spot",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by Septoria lycopersici. Small circular spots with dark borders and white/gray centers appear on lower leaves first. Spreads upward, causing severe defoliation.",
    treatment: ["Apply chlorothalonil or copper fungicides every 7–10 days in wet weather.", "Mulch soil to minimize rain splash from the soil.", "Remove severely infected lower leaves early in the season."],
    emoji: "🍅",
  },
  "Tomato___Spider_mites_Two-spotted_spider_mite": {
    displayName: "Spider Mite Damage",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by Tetranychus urticae. Tiny mites cause yellow stippling on leaf surfaces, fine webbing, and eventually bronzing and leaf drop. Population explosions occur in hot, dry conditions.",
    treatment: ["Apply miticides (abamectin, spiromesifen) or insecticidal soap.", "Increase humidity around plants; spider mites thrive in dry conditions.", "Introduce predatory mites (Phytoseiulus persimilis) for biological control."],
    emoji: "🍅",
  },
  "Tomato___Target_Spot": {
    displayName: "Target Spot",
    plant: "Tomato",
    severity: "medium",
    description: "Caused by Corynespora cassiicola. Brown lesions with concentric rings and yellow halos, resembling a target, appear on leaves, stems, and fruit. Common in tropical and subtropical regions.",
    treatment: ["Apply azoxystrobin or chlorothalonil fungicides at first signs.", "Improve air circulation through staking and pruning lower leaves.", "Avoid late-afternoon irrigation that leaves foliage wet overnight."],
    emoji: "🍅",
  },
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
    displayName: "Yellow Leaf Curl Virus",
    plant: "Tomato",
    severity: "critical",
    description: "A devastating viral disease transmitted by the whitefly (Bemisia tabaci). Infected plants show upward leaf curling, yellowing margins, stunted growth, and severe yield loss. No cure exists once infected.",
    treatment: ["⚠️ Control the whitefly vector immediately using systemic insecticides (imidacloprid).", "Use reflective mulch to deter whiteflies.", "Remove and destroy infected plants to reduce virus spread.", "Plant resistant varieties (TYLCV-resistant) in future seasons."],
    emoji: "🍅",
  },
  "Tomato___Tomato_mosaic_virus": {
    displayName: "Mosaic Virus",
    plant: "Tomato",
    severity: "high",
    description: "Caused by Tomato mosaic virus (ToMV). Symptoms include mottled light and dark green mosaic patterns on leaves, distortion, and reduced yield. Spreads mechanically through tools and hands.",
    treatment: ["There is no cure; remove and destroy infected plants.", "Disinfect all tools with 10% bleach or 70% alcohol between plants.", "Wash hands thoroughly before handling plants.", "Plant resistant varieties certified as ToMV-resistant."],
    emoji: "🍅",
  },
  "Tomato___healthy": {
    displayName: "Healthy",
    plant: "Tomato",
    severity: "none",
    description: "Your tomato plant is healthy! Leaves show good green color with no signs of disease, pests, or nutrient deficiency.",
    treatment: ["Continue consistent watering at the base to prevent foliar disease.", "Apply balanced fertilizer and monitor calcium levels to prevent blossom end rot."],
    emoji: "🍅",
  },
};

/**
 * Get formatted metadata for a model label string.
 * Falls back gracefully if the label isn't found.
 */
export function getDiseaseInfo(label) {
  const info = DISEASE_DB[label];
  if (!info) {
    const [plant, disease] = label.split("___");
    return {
      displayName: (disease || "Unknown").replace(/_/g, " "),
      plant: (plant || "Unknown").replace(/_/g, " "),
      severity: "unknown",
      description: "No additional information available for this diagnosis.",
      treatment: ["Consult a local agricultural extension office for advice."],
      emoji: "🌿",
    };
  }
  return info;
}

export const SEVERITY_CONFIG = {
  none:     { label: "Healthy",   color: "#22c55e", bg: "#dcfce7", ring: "#86efac" },
  low:      { label: "Low Risk",  color: "#84cc16", bg: "#ecfccb", ring: "#bef264" },
  medium:   { label: "Moderate",  color: "#f59e0b", bg: "#fef3c7", ring: "#fcd34d" },
  high:     { label: "Severe",    color: "#ef4444", bg: "#fee2e2", ring: "#fca5a5" },
  critical: { label: "Critical",  color: "#dc2626", bg: "#fef2f2", ring: "#f87171" },
  unknown:  { label: "Unknown",   color: "#6b7280", bg: "#f3f4f6", ring: "#d1d5db" },
};
