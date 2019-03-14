from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv

url = "https://www.tripadvisor.fr/Airlines"
html = urlopen(url)

soup = BeautifulSoup(html, 'lxml')
type(soup)

# Print out the text
text = soup.get_text()
#print(soup.text)

allDiv = soup.find_all('div', {'class':'wrapper'})
#allp = soup.find_all("p")

#print("Partie p")
#for i in allp:
 #  print(i.text)

print("Partie class wrapper")

f=open('CSVScrapper.csv','w')
f.write('avisClient')
f.write('\n')
for i in allDiv:
   #print(i.text)
    avis = i.text
    if (avis != ""):
       f.write(str(avis))
       f.write('\n')
f.close()

print("Fin chargement")
