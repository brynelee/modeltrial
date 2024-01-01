from server.knowledge_base.migrate import create_tables
from datetime import datetime
import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="please specify only one operate method once time.")

    parser.add_argument(
        "-r",
        "--recreate-vs",
        action="store_true",
        help=('''
            recreate vector store.
            use this option if you have copied document files to the content folder, but vector store has not been populated or DEFAUL_VS_TYPE/EMBEDDING_MODEL changed.
            '''
        )
    )
    parser.add_argument(
        "-u",
        "--update-in-db",
        action="store_true",
        help=('''
            update vector store for files exist in database.
            use this option if you want to recreate vectors for files exist in db and skip files exist in local folder only.
            '''
        )
    )
    parser.add_argument(
        "-i",
        "--increament",
        action="store_true",
        help=('''
            update vector store for files exist in local folder and not exist in database.
            use this option if you want to create vectors increamentally.
            '''
        )
    )
    parser.add_argument(
        "--prune-db",
        action="store_true",
        help=('''
            delete docs in database that not existed in local folder.
            it is used to delete database docs after user deleted some doc files in file browser
            '''
        )
    )
    parser.add_argument(
        "--prune-folder",
        action="store_true",
        help=('''
            delete doc files in local folder that not existed in database.
            is is used to free local disk space by delete unused doc files.
            '''
        )
    )
    parser.add_argument(
        "--kb-name",
        type=str,
        nargs="+",
        default=[],
        help=("specify knowledge base names to operate on. default is all folders exist in KB_ROOT_PATH.")
    )

    if len(sys.argv) <= 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        start_time = datetime.now()

        create_tables() 
        if args.recreate_vs:
            reset_tables()
            print("database talbes reseted")
            print("recreating all vector stores")
            folder2db(kb_names=args.kb_name, mode="recreate_vs")
        elif args.update_in_db:
            folder2db(kb_names=args.kb_name, mode="update_in_db")
        elif args.increament:
            folder2db(kb_names=args.kb_name, mode="increament")
        elif args.prune_db:
            prune_db_docs(args.kb_name)
        elif args.prune_folder:
            prune_folder_files(args.kb_name)

        end_time = datetime.now()
        print(f"总计用时： {end_time-start_time}")