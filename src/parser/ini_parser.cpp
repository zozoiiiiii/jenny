#include "ini_parser.h"
#include <stdio.h>
#include <assert.h>



IniParser::IniParser(const std::string& filename)  : m_strFileName(filename)
{}

IniParser::~IniParser()
{
    for (size_t i = 0; i < m_sections.size(); i++)
    {
        IniSection* pSection = m_sections[i];
        delete pSection;
    }
}


std::string IniParser::readLine(FILE *fp)
{
    if (feof(fp))
        return 0;

    char line[512] = { 0 };
    int nSize = sizeof(line) / sizeof(char);
    if (!fgets(line, nSize, fp))
    {
        free(line);
        return "";
    }

    // get one line
    int curr = strlen(line);
    if (line[curr - 1] == '\n')
    {
        line[curr - 1] = '\0';
        return line;
    }

    // it's a real long line
    char* pLongLine = nullptr;
    while ((line[curr - 1] != '\n') && !feof(fp))
    {
        if (pLongLine)
            free(pLongLine);

        if (curr == nSize - 1)
        {
            nSize *= 2;
            pLongLine = (char*)malloc(nSize * sizeof(char));
            if (!line)
            {
                return 0;
            }
        }

        fgets(pLongLine, nSize, fp);
        curr = strlen(line);
    }

    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

std::string IniParser::strip(const std::string& str)
{
    char* buf = (char*)malloc(str.length());
    strcpy(buf, str.c_str());
    
    size_t i;
    size_t len = strlen(buf);
    size_t offset = 0;
    for (i = 0; i < len; ++i)
    {
        char c = buf[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            ++offset;

        else
            buf[i - offset] = c;
    }
    buf[len - offset] = '\0';
    std::string result = buf;
    free(buf);
    return result;
}

bool IniParser::LoadFromFile()
{
    FILE *file = fopen(m_strFileName.c_str(), "r");
    if (file == 0)
        return false;

    IniSection* pSection = nullptr;
    while (true)
    {
        std::string line = readLine(file);
        if (line.empty())
            break;

        line = strip(line);
        char firstChar = line.at(0);
        switch (firstChar)
        {
        case '[':
            pSection = new IniSection;
            pSection->name = line;
            if (line.back() == ']')
                pSection->name = std::string(line.c_str() + 1, line.length() - 2);

            m_sections.push_back(pSection);
            break;
        case '\0':
        case '#':
        case ';':
            break;
        default:
            if (!pSection)
                break;

            int pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                pSection->values[key] = value;
            }
            break;
        }
    }
    fclose(file);
    return true;
}

int IniParser::GetSectionCount() const
{
    return m_sections.size();
}

std::string IniParser::GetSectionByIndex(int index) const
{
    if (index < 0 || index >= m_sections.size())
        return "";

    return m_sections[index]->name;
}


int IniParser::GetSectionItemCount(int sect_index) const
{
    if (sect_index < 0 || sect_index >= m_sections.size())
        return -1;

    IniSection* pSection = m_sections[sect_index];
    return pSection->values.size();
}

bool IniParser::GetSectionItem(int sect_index, int item_index, std::string& key, std::string& value) const
{
    if (sect_index < 0 || sect_index >= m_sections.size())
        return false;

    IniSection* pSection = m_sections[sect_index];
    if (item_index < 0 || item_index >= pSection->values.size())
        return false;

    auto itor = pSection->values.begin();
    std::advance(itor, item_index);
    key = itor->first;
    value = itor->second;
    return true;
}


int IniParser::ReadInteger(size_t sect_index, const std::string& key, int def) const
{
    if (sect_index < 0 || sect_index >= m_sections.size())
        return def;

    IniSection* pSection = m_sections[sect_index];
    auto itor = pSection->values.find(key);
    if (itor == pSection->values.end())
        return def;

    std::string value = itor->second;
    return atoi(value.c_str());
}

std::string IniParser::ReadString(size_t sect_index, const std::string& key, const std::string& def) const
{
    if (sect_index < 0 || sect_index >= m_sections.size())
        return def;

    IniSection* pSection = m_sections[sect_index];
    auto itor = pSection->values.find(key);
    if (itor == pSection->values.end())
        return def;

    return itor->second;
}

float IniParser::ReadFloat(size_t sect_index, const std::string& key, float def) const
{
    if (sect_index < 0 || sect_index >= m_sections.size())
        return def;

    IniSection* pSection = m_sections[sect_index];
    auto itor = pSection->values.find(key);
    if (itor == pSection->values.end())
        return def;

    std::string value = itor->second;
    return (float)atof(value.c_str());
}
