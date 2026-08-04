import wikipediaapi

def sectionRecursion(sections, lvl=0, sstr=""): 
    outputs = []
    for section in sections: 
        section_str = sstr + f": {section.title}"
        outputs.append((lvl, section_str[2:], section.text))
        outputs += sectionRecursion(section.sections, lvl+1, section_str)
    return outputs

def page2md(wikipage): 
    url = wikipage.fullurl
    title = wikipage.title
    summary = wikipage.summary
    content = wikipage.sections
    n = "\n\n"
    md = ""
    md += f"## {title}\n"
    md += f"# Summary\n" 
    md += summary + n
    sections = sectionRecursion(content)
    for section in sections: 
        if(section[1] == "See also"): break
        md += f"# {section[1]}\n" 
        md += section[2].replace("\n", " ").replace("  ", " ") + n
    return md, url[30:]

def title2md(wiki_title, wikiapi): 
    page =  wikiapi.page(wiki_title)
    md, name = page2md(page)
    with open(f"./documents/{name}.md", "w", encoding="utf-8") as f: f.write(md)




if __name__ == "__main__": 
    wiki = wikipediaapi.Wikipedia(
        user_agent='AIResearchBot/1.0 (your_email@example.com)',
        language='en'
    )

    senators = ['Tommy Tuberville', 'Katie Britt', 'Lisa Murkowski', 'Dan Sullivan', 'Mark Kelly', 'Ruben Gallego', 'John Boozman', 'Tom Cotton', 
                'Alex Padilla', 'Adam Schiff', 'Michael Bennet', 'John Hickenlooper', 'Richard Blumenthal', 'Chris Murphy', 'Chris Coons', 
                'Lisa Blunt Rochester', 'Rick Scott', 'Ashley Moody', 'Jon Ossoff', 'Raphael Warnock', 'Brian Schatz', 'Mazie Hirono', 'Mike Crapo', 
                'Jim Risch', 'Dick Durbin', 'Tammy Duckworth', 'Todd Young', 'Jim Banks', 'Chuck Grassley', 'Joni Ernst', 'Jerry Moran', 'Roger Marshall', 
                'Mitch McConnell', 'Rand Paul', 'Bill Cassidy', 'John Kennedy', 'Susan Collins', 'Angus King', 'Chris Van Hollen', 'Angela Alsobrooks', 
                'Elizabeth Warren', 'Ed Markey', 'Gary Peters', 'Elissa Slotkin', 'Amy Klobuchar', 'Tina Smith', 'Roger Wicker', 'Cindy Hyde-Smith', 
                'Josh Hawley', 'Eric Schmitt', 'Steve Daines', 'Tim Sheehy', 'Deb Fischer', 'Pete Ricketts', 'Catherine Cortez Masto', 'Jacky Rosen', 
                'Jeanne Shaheen', 'Maggie Hassan', 'Cory Booker', 'Andy Kim', 'Martin Heinrich', 'Ben Ray Luján', 'Chuck Schumer', 'Kirsten Gillibrand', 
                'Thom Tillis', 'Ted Budd', 'John Hoeven', 'Kevin Cramer', 'Bernie Moreno', 'Jon Husted', 'James Lankford', 'Alan Armstrong', 'Ron Wyden', 
                'Jeff Merkley', 'John Fetterman', 'Dave McCormick', 'Jack Reed', 'Sheldon Whitehouse', 'Tim Scott', 'Darline Graham', 'John Thune', 
                'Mike Rounds', 'Marsha Blackburn', 'Bill Hagerty', 'John Cornyn', 'Ted Cruz', 'Mike Lee', 'John Curtis', 'Bernie Sanders', 'Peter Welch', 
                'Mark Warner', 'Tim Kaine', 'Patty Murray', 'Maria Cantwell', 'Shelley Moore Capito', 'Jim Justice', 'Ron Johnson', 'Tammy Baldwin', 
                'John Barrasso', 'Cynthia Lummis']

    from tqdm import tqdm
    for senator in tqdm(senators[2:]): 
        title2md(senator, wiki)
    