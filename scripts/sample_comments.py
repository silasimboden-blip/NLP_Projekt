"""20 hand-written German news comments, 5 per label, for the load generator.

The label tag is for sampling balance only — it is NOT sent to the service.
The classifier is genuinely zero-shot and only sees the comment text.
"""

SAMPLES = [
    # Zustimmung
    ("Zustimmung", "Sehr guter, gut recherchierter Artikel, danke!"),
    ("Zustimmung", "Endlich mal eine ausgewogene Berichterstattung zu diesem Thema."),
    ("Zustimmung", "Genau meine Meinung, vielen Dank für diesen Beitrag."),
    ("Zustimmung", "Ich finde es wichtig, dass darüber berichtet wird."),
    ("Zustimmung", "Klar formuliert und auf den Punkt gebracht, top!"),

    # sachliche Kritik
    ("sachliche Kritik", "Mir fehlen hier Quellenangaben für die zitierten Zahlen."),
    ("sachliche Kritik", "Die Studie wird zitiert, aber nicht verlinkt — schade."),
    ("sachliche Kritik", "Der Artikel berücksichtigt die Gegenposition leider nicht."),
    ("sachliche Kritik", "Einige Aussagen sind zu allgemein und sollten präzisiert werden."),
    ("sachliche Kritik", "Ich hätte mir mehr Hintergrund zu den Folgen gewünscht."),

    # Empörung
    ("Empörung", "Das ist eine absolute Frechheit, wer hat das durchgewunken??"),
    ("Empörung", "Unfassbar, dass so etwas immer noch passieren kann!"),
    ("Empörung", "Eine bodenlose Sauerei, die sofort gestoppt werden muss."),
    ("Empörung", "Das macht mich richtig wütend, wie kann das sein?"),
    ("Empörung", "Skandal! Sofort Konsequenzen für die Verantwortlichen!"),

    # Beleidigung oder persönlicher Angriff
    ("Beleidigung oder persönlicher Angriff", "Der Autor dieses Texts ist offensichtlich ein Vollidiot."),
    ("Beleidigung oder persönlicher Angriff", "Wer so etwas schreibt, hat doch keine Ahnung von gar nichts."),
    ("Beleidigung oder persönlicher Angriff", "Typisch dieser dumme Journalist, immer dasselbe Geschwätz."),
    ("Beleidigung oder persönlicher Angriff", "@nutzer123: du bist echt zu blöd, das zu verstehen."),
    ("Beleidigung oder persönlicher Angriff", "Was ein Schwachsinn, der Redakteur sollte gefeuert werden."),
]
