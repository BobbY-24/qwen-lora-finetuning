# Selected qualitative cases

These 14 cases were selected after viewing fixed predictions to illustrate failure mechanisms and metric limitations. Labels are **assistant qualitative judgments**, not independent human gold, and must not be aggregated into a semantic-accuracy estimate. A human reviewer should validate them blind to model identity.

Grades: 0 wrong, 1 partial, 2 contextually adequate, null unscorable reference.

## JK: improvement

ID: `hf-00130-d8b7b62132`. Usage: You’re fired! JK, I could never fire you.

Reference: Just kidding

Base: In this context, "JK" likely refers to a person's initials or nickname. The phrase "I could never fire you" suggests that the speaker is expressing regret for firing someone, possibly due to a misunderstanding or miscommunication.

LoRA: Just kidding

Judgment (0 → 2): The base invents initials/nickname and regret; LoRA correctly recognizes the joking reversal.

## 1CE: improvement

ID: `hf-00254-fb0ddda9d8`. Usage: I’ve only been there 1CE, but I loved it.

Reference: Once

Base: In this context, "1CE" likely stands for "First Class Experience," referring to a highly satisfactory or enjoyable experience that is exceptional or top-notch. The phrase indicates that the speaker had an excellent first-class experience at the location they were visiting.

LoRA: Once

Judgment (0 → 2): First Class Experience does not fit the frequency slot; Once does.

## QQ: improvement

ID: `hf-01344-131a537650`. Usage: Hey, QQ: What time does the event start?

Reference: Quick question

Base: In this context, "QQ" likely refers to a messaging app or service that allows users to send messages and communicate with others. The phrase "What time does the event start?" is asking for information about the start time of an event using the messaging app.

LoRA: Quick question

Judgment (0 → 2): A messaging-app interpretation misses the question marker; Quick question fits the utterance.

## VWP: improvement

ID: `hf-01626-66fe3ac1e1`. Usage: You outsmarted me in that game, VWP!

Reference: Very well played

Base: In this context, "VWP" likely stands for "Victory Point," which is a term used in video games to indicate the number of points a player has gained after completing a level or mission. The phrase "outsmarted me in that game, VWP!" suggests that the player achieved victory by defeating an

LoRA: Very well played

Judgment (0 → 2): Victory Point misses praise for the opponent; Very well played fits.

## G.O.A.T: both_fail

ID: `hf-00008-2587f51157`. Usage: Michael Jordan is the GOAT of basketball.

Reference: The greatest of all time

Base: The phrase "GOAT" refers to the greatest overall athlete (G) in a particular sport (O), which is often used to describe an individual who has been selected for multiple major awards or honors in their field. In this context, Michael Jordan is being referred to as the Greatest Overall Athlete in basketball, which

LoRA: Good old man of all time

Judgment (0 → 0): Both expansions are wrong. LoRA shares several words with the reference despite inventing Good old man.

## Rent free: both_fail

ID: `hf-00018-e2dd8db6da`. Usage: That song is living rent-free in my head

Reference: To be constantly thinking or upset about something.

Base: The phrase "rent free" means that the song is available for personal use without any obligation to pay rent or other financial costs associated with ownership.

LoRA: A phrase used to describe a song that has been played many times but never become part of your memory.

Judgment (0 → 0): The base treats this as financial rent; LoRA says never become part of memory, contradicting the intrusive-memory sense.

## Karen: both_fail

ID: `hf-00093-1829a56b9c`. Usage: That woman asked to speak to the manager, total Karen move.

Reference: A stereotype for women who behave extremely unreasonably and childish – especially towards service staff or people of colour – and tend to invoke their racial privilege

Base: In this context, "Karen" refers to a female employee who has been transferred or reassigned from her previous role at the company. The phrase "total Karen move" suggests that she is being replaced by another person or department within the organization.

LoRA: Korean

Judgment (0 → 0): Neither an employee reassignment nor Korean explains the manager-demanding stereotype.

## Slay: both_fail

ID: `hf-00074-77daff0953`. Usage: She’s slaying in that outfit tonight!

Reference: To do something well.

Base: The phrase "She's slaying in that outfit tonight!" is an idiomatic expression used to describe someone who is actively engaging or participating in a particular activity or event. In this context, it suggests that the person is actively participating in some form of combat or battle, likely involving weapons or equipment. The use of "

LoRA: To describe someone who is wearing a costume or dress

Judgment (0 → 0): Combat is wrong here. Merely wearing a costume also misses doing/looking exceptionally well.

## e-girl: partial_to_wrong

ID: `hf-00026-6a3c249c2b`. Usage: She’s such an e-girl with her neon eyeliner and platform boots.

Reference: similar to emo or goth culture, but they use the internet to express themselves.

Base: The term "e-girl" refers to someone who is very fashionable or trendy, often wearing bright colors like neon colors, high heels, and other fashion accessories. The phrase suggests that the person has a certain style or look that others find attractive.

LoRA: Elegant

Judgment (1 → 0): The base captures a fashion aspect but misses internet subculture; Elegant is not an explanation of e-girl.

## 4AO: partial_to_wrong

ID: `hf-00272-29d14498bf`. Usage: That movie is rated 4AO.

Reference: For adults only

Base: The term "4AO" refers to a rating system used by some streaming services or platforms for movies. It stands for "Adult Online Rating," which indicates that the movie contains mature content suitable only for adults.

LoRA: Above average

Judgment (1 → 0): The base captures adult-only content but invents the expansion and platform claim; Above average loses the meaning entirely.

## ADMINR: metric_artifact

ID: `hf-00331-448fc3b8e3`. Usage: She’s the ADMINR for this forum.

Reference: Administrator

Base: In this context, "ADMINR" refers to a role or position within an organization or system that is responsible for managing administrative tasks related to forums or similar platforms.

LoRA: Administrator

Judgment (2 → 2): The base correctly describes a forum administrator but receives zero lexical F1 because the target is the single word Administrator. LoRA exact-matches it. A +1 lexical jump is not a semantic improvement here.

## TBH: metric_artifact

ID: `hf-00104-ed5feb1667`. Usage: TBH, I didn’t enjoy the party last night.

Reference: short for “to be honest” and is usually used when making a comment or a joke (best to stick to texts with this one too).

Base: In this context, "TBH" stands for "To be honest," indicating that the speaker is admitting they did not enjoy the party last night.

LoRA: To tell the truth

Judgment (2 → 2): To tell the truth is an adequate contextual paraphrase of To be honest, despite a lower overlap score than the longer base response.

## IMO: loss_of_specificity

ID: `hf-00177-e9b63402ae`. Usage: IMO, that’s the best coffee shop in town.

Reference: In My Opinion

Base: IMO stands for "In My Opinion," indicating a subjective opinion or recommendation. In this context, it suggests that the speaker believes the coffee shop is the best in town based on their personal judgment.

LoRA: I think so

Judgment (2 → 1): The base explicitly marks a subjective opinion. I think so conveys subjectivity but is a weaker substitute and not the acronym expansion; do not force this into an unequivocal correct-to-wrong label.

## I can’t even: reference_review

ID: `hf-00090-acd25f533a`. Usage: Ion even care what they think.

Reference: an expression commonly used when someone is feeling either so overjoyed or exasperated that they can’t even find the right words; depending on the context.

Base: "I can't even" means "I don't have the ability to do something." In this context, it suggests that someone is unable or unwilling to perform a task or action.

LoRA: Even if

Judgment (None → None): The supplied term is I can’t even, but the usage is Ion even care. This needs reference adjudication; do not silently repair the gold or count it as a clean semantic test.

## Unassigned claims

No unambiguous fully-correct-to-wrong slang example was assigned in this small selected review; partial-to-wrong and loss-of-specificity cases are labeled separately. No memorization or excessive-slang-style claim is established. Term disjointness does not rule out pretraining familiarity. Human annotation of the full or a prespecified random sample is still needed.
