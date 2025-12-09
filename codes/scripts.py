import json
import os

# --- Definição dos Dados ---
# Cada item no dicionário representa o conteúdo de um arquivo JSON.
# A chave é o nome do arquivo e o valor é o conteúdo em formato de dicionário Python.
json_data = {
    # 10 Primeiros Arquivos
    "Agenda.json": {
      "_id": { "$oid": "Agenda01" },
      "status": "cheia",
      "data": "2000-05-18T16:00:00Z"
    },
    "Assina.json": {
      "_id": { "$oid": "Assina01" },
      "paciente_id": "UsuarioPaciente01",
      "plano_id": "Plano01"
    },
    "Consulta.json": {
      "_id": { "$oid": "Consulta01" },
      "medico_id": "UsuarioMedico01",
      "paciente_id": "UsuarioPaciente01",
      "status": "concluida",
      "diagnostico": "gripe",
      "data": "2000-05-17T16:00:00Z",
      "preescricao_id": "Preescricao01"
    },
    "Especialidade.json": {
      "_id": { "$oid": "Especialidade01" }, # Corrigido de "Especializado01" para ser unico
      "nome" : "neuro",
      "descricao": "especializado doenças relacionadas ao cérebro",
      "cod_cbo": "225112"
    },
    "Especializado.json": {
      "_id": { "$oid": "Especializado01" },
      "medico_id": "UsuarioMedico01",
      "especialidade_id": "Especialidade01"
    },
    "Exame.json": {
      "_id": { "$oid": "Exame01" },
      "tipo": "espirometria",
      "data": "2000-05-18T16:00:00Z",
      "resultado": "gripe",
      "preescricao_id": "Preescricao01",
      "historico_id": "Historico01"
    },
    "Frequenta.json": {
      "_id": { "$oid": "Frequenta01" },
      "usuario_id": "Usuario01",
      "unidade_id": "Unidade01"
    },
    "HistoricoPaciente.json": {
      "_id": { "$oid": "Historico01" },
      "data_entrada": "2000-05-18T16:00:00Z",
      "descricao": "paciente01 diagnosticado com gripe, recebeu preescricao com...",
      "paciente_id": "UsuarioPaciente01"
    },
    "MarcaAgenda.json": {
      "_id": { "$oid": "MarcaAgenda01" },
      "recepcionista_id": "UsuarioRecepcionista01",
      "agenda_id": "Agenda01"
    },
    "MarcaPaciente.json": {
      "_id": { "$oid": "MarcaPaciente01" },
      "recepcionista_id": "UsuarioRecepcionista01",
      "paciente_id": "UsuarioPaciente01"
    },

    # 10 Últimos Arquivos
    "Pagamento.json": {
      "_id": { "$oid": "Pagamento01" },
      "paciente_id": "UsuarioPaciente01",
      "valor": 59.90,
      "forma_pagamento": "debito",
      "data": "2000-05-18T16:00:00Z",
      "consulta_id": "Consulta01"
    },
    "PlanoDeSaude.json": {
      "_id": { "$oid": "Plano01" },
      "nome": "rapvide",
      "tipo": "afiliado",
      "cobertura": "Toda Sergipe"
    },
    "Preescricao.json": {
      "_id": { "$oid": "Preescricao01" }
    },
    "Procedimento.json": {
      "_id": { "$oid": "Procedimento01" },
      "descricao": "procedimento simples na regiao pulmonar",
      "preescricao_id": "Preescricao01",
      "historico_id": "Historico01"
    },
    "Remedio.json": {
      "_id": { "$oid": "Paracetamol" },
      "miligramas": 500,
      "tarja": None, # Em JSON é null, em Python é None
      "preescricao_id": "Preescricao01",
      "historico_id": "Historico01"
    },
    "UnidadeDeAtendimento.json": {
      "_id": { "$oid": "Unidade01" },
      "numero": 0,
      "bairro": "Rosa Elze",
      "rua": "Avenida Marechal Rondon",
      "cidade": "Sao cristovao",
      "CEP": "49100-000"
    },
    "UsuarioAdm.json": {
      "_id": { "$oid": "UsuarioAdm01" },
      "email": "emailg@gmail.com",
      "senha": "senha12",
      "nome": "Roberto da Silva Reis",
      "telefone": "40028923",
      "data_nascimento": { "$date": "2000-02-01T00:00:00Z" },
      "tipo_usuario": "administrador"
    },
    "UsuarioMedico.json": {
      "_id": { "$oid": "UsuarioMedico01" },
      "email": "emailm@gmail.com",
      "senha": "senha125",
      "telefone": "40028925",
      "nome": "José Diguidin Xaulin",
      "data_nascimento": { "$date": "2000-05-01T00:00:00Z" },
      "tipo_usuario": "medico",
      "medico": {
        "crm": 10000,
        "disponibilidade": "segunda a sexta",
        "agenda_id": "Agenda01"
      }
    },
    "UsuarioPaciente.json": {
      "_id": { "$oid": "UsuarioPaciente01" },
      "email": "emailf@gmail.com",
      "senha": "senha123",
      "telefone": "40028922",
      "nome": "Tohdo Enty",
      "data_nascimento": { "$date": "2000-01-01T00:00:00Z" },
      "tipo_usuario": "paciente",
      "paciente": {
        "prioridade": 2
      }
    },
    "UsuarioRecepcionista.json": {
      "_id": { "$oid": "UsuarioRecepcionista01" },
      "email": "emailh@gmail.com",
      "senha": "senha124",
      "telefone": "40028924",
      "nome": "Tia Tendo",
      "data_nascimento": { "$date": "2000-03-01T00:00:00Z" },
      "tipo_usuario": "recepcionista",
      "recepcionista": {
        "consulta_id": "Consulta01"
      }
    }
}

# --- Lógica para Criar os Arquivos ---

# Nome do diretório onde os arquivos serão salvos
output_dir = "json_output"

# Cria o diretório se ele não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Itera sobre cada item no dicionário de dados
for filename, content in json_data.items():
    # Monta o caminho completo do arquivo (diretório + nome do arquivo)
    file_path = os.path.join(output_dir, filename)

    # Abre o arquivo no modo de escrita ('w')
    with open(file_path, 'w', encoding='utf-8') as f:
        # Escreve o conteúdo do dicionário no arquivo, formatado como JSON
        # indent=2 cria uma formatação legível
        # ensure_ascii=False garante que caracteres como 'ç' e 'ã' sejam salvos corretamente
        json.dump(content, f, indent=2, ensure_ascii=False)

print(f"Sucesso! {len(json_data)} arquivos JSON foram gerados no diretório '{output_dir}'.")