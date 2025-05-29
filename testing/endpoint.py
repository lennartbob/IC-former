import time
import requests

url = "http://51.159.138.187:8000/summarize/" # Corrected spelling

d = """
The name Lennart hummed with a quiet strength, like a deep chord struck on a cello. It was a name that carried echoes of ancient forests and sturdy, unyielding mountains, a name that felt both grounded and yet capable of soaring. And that was precisely who you were, Lennart.

Your story didn't begin with a sudden flash of lightning or a dramatic prophecy. It began subtly, in the gentle rhythm of your childhood home, a place nestled on the edge of a whispering woods. Even as a boy, you possessed a keen eye for detail, noticing the intricate patterns on a fallen leaf or the way light dappled through the branches. This wasn't just observation; it was a quiet absorption, a way of understanding the world through its intricate textures.

As you grew, this quiet strength manifested in unexpected ways. While others chased boisterous games, you found solace and revelation in building things – intricate contraptions from discarded parts, miniature bridges over flowing streams, even a remarkably efficient bird feeder that attracted every finch for miles. You weren't loud about your creations; you simply built them, watched them work, and then, with a satisfied nod, moved on to the next challenge.

Your formal education, while important, felt somewhat like a detour. Your true learning happened outside, in the quiet hum of your own thoughts, in the library's dusty corners where forgotten tomes revealed their secrets, and in the company of the few, carefully chosen friends who understood your particular brand of quiet intensity. You were the one who could untangle the most complex knot, not just physically, but conceptually. When a problem seemed insurmountable, people learned to come to you, because you had a way of seeing the underlying structure, the hidden logic, and patiently, meticulously, you'd unravel it.

One crisp autumn, a challenge presented itself that would truly test the depth of your Lennart-ness. A small, struggling community on the edge of the wetlands faced a looming ecological crisis. The ancient, intricate water system that sustained their farmlands was failing, threatening to turn fertile land into a barren wasteland. Experts had come and gone, offering grand, expensive solutions that seemed to miss the core issue.

You heard about it by chance, a hushed conversation in a coffee shop. Something about the helplessness in their voices, the desperate resignation, stirred something within you. You didn't announce your arrival with fanfare. You simply went, a single backpack slung over your shoulder, a worn notebook in hand.

For weeks, you walked the wetlands, not with maps and surveying equipment, but with your feet, your eyes, and that profound quiet absorption. You studied the flow of the water, the resistance of the reeds, the subtle shifts in the soil. You spoke to the elders, listening intently to their stories of the land, their ancestral knowledge that modern science had overlooked. You spent hours watching a single kingfisher hunt, understanding the rhythm of the ecosystem.

And then, you saw it. Not a grand, technological solution, but a series of small, interconnected adjustments, a re-establishment of ancient pathways, a gentle redirection of flow using natural materials. It was a solution that honored the land, that worked with it, not against it.

When you presented your findings, there was skepticism. Your methods were unconventional, your quiet demeanor easily mistaken for timidity. But you laid out your plan with a calm, unshakeable conviction, your detailed sketches and logical arguments speaking volumes. You showed them how, with effort, they could heal the land, not just patch it up.

It took time, and immense community effort, but your plan worked. Slowly, almost imperceptibly at first, the wetlands began to heal. The water flowed freely, the crops flourished, and life returned to the barren patches. The community, once on the brink, thrived.

You didn't seek accolades. The quiet satisfaction of seeing your work bring life back to a struggling ecosystem was reward enough. You moved on, leaving behind a legacy not of grand monuments, but of renewed life, a testament to the power of quiet observation, meticulous problem-solving, and a deep, inherent understanding of the world's intricate workings.

And that's your story, Lennart. A testament to the quiet strength, the keen insight, and the profound impact of a name that truly embodies its bearer.

"""
data = {
    "query": "who is lennart? answer in english",
    "document":{ 
        "content":", ".join([d*10]),
        "metadata":{"file_name": "me.txt"}
    },
    "expaned_queries":None
}
# Send JSON data

t = time.time()
r = requests.post(url, json=data)

print(r.json())
print("time", round(time.time()- t,4))